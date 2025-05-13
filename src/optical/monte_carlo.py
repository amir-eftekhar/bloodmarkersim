#!/usr/bin/env python3
"""
Monte Carlo simulation of light transport in the wrist tissue.
Simulates photon propagation through multi-layered tissue including blood vessels.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
from pathlib import Path

# Add the parent directory to the path to import utility modules
sys.path.append(str(Path(__file__).parent.parent))
from utils.constants import (ABSORPTION_GREEN, ABSORPTION_RED, ABSORPTION_NIR,
                           SCATTERING_GREEN, SCATTERING_RED, SCATTERING_NIR,
                           ANISOTROPY, REFRACTIVE_INDEX, MONTE_CARLO_PHOTONS,
                           TISSUE_GRID_RESOLUTION, MAX_SIMULATION_DEPTH)


class PhotonPacket:
    """
    Class representing a photon packet in the Monte Carlo simulation.
    """
    
    def __init__(self, position, direction, weight=1.0, wavelength='red'):
        """
        Initialize a photon packet.
        
        Args:
            position (numpy.ndarray): 3D position vector [x, y, z].
            direction (numpy.ndarray): 3D direction vector [dx, dy, dz].
            weight (float): Photon packet weight (intensity).
            wavelength (str): Wavelength band ('green', 'red', or 'nir').
        """
        self.position = np.array(position, dtype=float)
        self.direction = np.array(direction, dtype=float) / np.linalg.norm(direction)
        self.weight = weight
        self.wavelength = wavelength
        self.path = [self.position.copy()]  # Track photon path
        self.alive = True
        self.layer = None  # Current tissue layer
        self.step_size = 0.0
        self.total_distance = 0.0
        
    def move(self):
        """Move the photon packet by the current step size."""
        self.position += self.direction * self.step_size
        self.path.append(self.position.copy())
        self.total_distance += self.step_size
        
    def scatter(self, g):
        """
        Scatter the photon in a new direction according to the Henyey-Greenstein phase function.
        
        Args:
            g (float): Anisotropy factor.
        """
        if abs(g) < 1e-10:
            # Isotropic scattering
            cos_theta = 2 * np.random.random() - 1
        else:
            # Henyey-Greenstein phase function
            temp = (1 - g * g) / (1 - g + 2 * g * np.random.random())
            cos_theta = (1 + g * g - temp * temp) / (2 * g)
            
            # Handle numerical error
            if cos_theta < -1:
                cos_theta = -1
            elif cos_theta > 1:
                cos_theta = 1
        
        # Azimuthal angle
        phi = 2 * np.pi * np.random.random()
        
        # Calculate new direction
        if abs(self.direction[2]) > 0.99999:
            # Special case for photons traveling close to the z-axis
            sin_theta = np.sqrt(1 - cos_theta * cos_theta)
            ux = sin_theta * np.cos(phi)
            uy = sin_theta * np.sin(phi)
            uz = np.sign(self.direction[2]) * cos_theta
            self.direction = np.array([ux, uy, uz])
        else:
            # General case
            sin_theta = np.sqrt(1 - cos_theta * cos_theta)
            temp = np.sqrt(1 - self.direction[2] * self.direction[2])
            
            # Rotate direction vector
            ux = sin_theta * (self.direction[0] * self.direction[2] * np.cos(phi) - 
                             self.direction[1] * np.sin(phi)) / temp + self.direction[0] * cos_theta
            
            uy = sin_theta * (self.direction[1] * self.direction[2] * np.cos(phi) + 
                             self.direction[0] * np.sin(phi)) / temp + self.direction[1] * cos_theta
            
            uz = -sin_theta * cos_phi * temp + self.direction[2] * cos_theta
            
            self.direction = np.array([ux, uy, uz])
            self.direction /= np.linalg.norm(self.direction)


class MonteCarloSimulation:
    """
    Class for Monte Carlo simulation of light transport in tissue.
    """
    
    def __init__(self, tissue_model, config=None):
        """
        Initialize the Monte Carlo simulation.
        
        Args:
            tissue_model (dict): 3D tissue model containing tissue types.
            config (dict, optional): Configuration parameters for the simulation.
        """
        self.tissue_model = tissue_model
        self.config = config or self._default_config()
        
        # Tissue grid properties
        self.x = tissue_model['x']
        self.y = tissue_model['y']
        self.z = tissue_model['z']
        self.tissue_types = tissue_model['tissue_types']
        
        # Initialize optical properties based on wavelength
        self.set_wavelength(self.config['wavelength'])
        
        # Detection grid - record photon exit positions and weights
        self.detection_grid = np.zeros((len(self.y), len(self.x)))
        self.detected_photons = []
        
        # Absorption map - record where photons are absorbed
        self.absorption_map = np.zeros_like(self.tissue_types, dtype=float)
    
    def _default_config(self):
        """Default configuration for the Monte Carlo simulation."""
        return {
            'num_photons': MONTE_CARLO_PHOTONS,
            'wavelength': 'red',  # 'green', 'red', or 'nir'
            'source_position': [0, 0, 0],  # [x, y, z] in cm
            'source_direction': [0, 0, 1],  # Initial direction vector
            'source_radius': 0.1,  # Source radius in cm
            'detector_radius': 0.2,  # Detector radius in cm
            'detector_position': [0.5, 0, 0],  # Center of detector in cm
            'max_weight': 1e-4,  # Minimum photon weight before termination
            'max_step_size': 0.1,  # Maximum step size in cm
            'detection_angle': np.pi/2,  # Max angle for detection in radians
            'record_paths': False  # Whether to record full photon paths
        }
    
    def set_wavelength(self, wavelength):
        """
        Set the simulation wavelength and corresponding optical properties.
        
        Args:
            wavelength (str): Wavelength band ('green', 'red', or 'nir').
        """
        self.wavelength = wavelength
        
        # Set absorption and scattering coefficients based on wavelength
        if wavelength == 'green':
            self.absorption = ABSORPTION_GREEN
            self.scattering = SCATTERING_GREEN
        elif wavelength == 'red':
            self.absorption = ABSORPTION_RED
            self.scattering = SCATTERING_RED
        elif wavelength == 'nir':
            self.absorption = ABSORPTION_NIR
            self.scattering = SCATTERING_NIR
        else:
            raise ValueError(f"Unknown wavelength: {wavelength}")
        
        # Create lookup tables for optical properties by tissue type
        self.absorption_coeff = np.zeros(7)  # 7 tissue types (including outside)
        self.scattering_coeff = np.zeros(7)
        self.anisotropy = np.zeros(7)
        self.refractive_idx = np.zeros(7)
        
        # Tissue type mapping: 0=outside, 1=epidermis, 2=dermis, 3=fat, 4=muscle, 5=blood, 6=bone
        
        # Absorption coefficients
        self.absorption_coeff[0] = 0.0  # Outside (air)
        self.absorption_coeff[1] = self.absorption['epidermis']
        self.absorption_coeff[2] = self.absorption['dermis']
        self.absorption_coeff[3] = self.absorption['fat']
        self.absorption_coeff[4] = self.absorption['muscle']
        self.absorption_coeff[5] = 0.75 * self.absorption['oxyhemoglobin'] + 0.25 * self.absorption['deoxyhemoglobin']  # Blood (75% oxygenated)
        self.absorption_coeff[6] = 0.1  # Bone (approximate)
        
        # Scattering coefficients
        self.scattering_coeff[0] = 0.0  # Outside (air)
        self.scattering_coeff[1] = self.scattering['epidermis']
        self.scattering_coeff[2] = self.scattering['dermis']
        self.scattering_coeff[3] = self.scattering['fat']
        self.scattering_coeff[4] = self.scattering['muscle']
        self.scattering_coeff[5] = self.scattering['blood']
        self.scattering_coeff[6] = 30.0  # Bone (approximate)
        
        # Anisotropy factors
        self.anisotropy[0] = 0.0  # Outside (air)
        self.anisotropy[1] = ANISOTROPY['epidermis']
        self.anisotropy[2] = ANISOTROPY['dermis']
        self.anisotropy[3] = ANISOTROPY['fat']
        self.anisotropy[4] = ANISOTROPY['muscle']
        self.anisotropy[5] = ANISOTROPY['blood']
        self.anisotropy[6] = 0.9  # Bone (approximate)
        
        # Refractive indices
        self.refractive_idx[0] = REFRACTIVE_INDEX['air']
        self.refractive_idx[1] = REFRACTIVE_INDEX['epidermis']
        self.refractive_idx[2] = REFRACTIVE_INDEX['dermis']
        self.refractive_idx[3] = REFRACTIVE_INDEX['fat']
        self.refractive_idx[4] = REFRACTIVE_INDEX['muscle']
        self.refractive_idx[5] = REFRACTIVE_INDEX['blood']
        self.refractive_idx[6] = 1.55  # Bone (approximate)
    
    def get_tissue_type(self, position):
        """
        Get the tissue type at a given position.
        
        Args:
            position (numpy.ndarray): 3D position vector [x, y, z].
            
        Returns:
            int: Tissue type at the position, or 0 if outside the grid.
        """
        # Check if position is within grid bounds
        x, y, z = position
        
        # Get indices in the grid
        ix = np.searchsorted(self.x, x) - 1
        iy = np.searchsorted(self.y, y) - 1
        iz = np.searchsorted(self.z, z) - 1
        
        # Check if within bounds
        if (ix < 0 or ix >= len(self.x)-1 or
            iy < 0 or iy >= len(self.y)-1 or
            iz < 0 or iz >= len(self.z)-1):
            return 0  # Outside the grid
        
        return self.tissue_types[ix, iy, iz]
    
    def get_optical_properties(self, position):
        """
        Get optical properties at a given position.
        
        Args:
            position (numpy.ndarray): 3D position vector [x, y, z].
            
        Returns:
            tuple: (absorption coefficient, scattering coefficient, anisotropy, refractive index)
        """
        tissue_type = self.get_tissue_type(position)
        
        return (self.absorption_coeff[tissue_type],
                self.scattering_coeff[tissue_type],
                self.anisotropy[tissue_type],
                self.refractive_idx[tissue_type])
    
    def launch_photon(self):
        """
        Launch a new photon packet with random initial position within the source area.
        
        Returns:
            PhotonPacket: A new photon packet.
        """
        # Random position within the source area
        r = self.config['source_radius'] * np.sqrt(np.random.random())
        theta = 2 * np.pi * np.random.random()
        
        x = self.config['source_position'][0] + r * np.cos(theta)
        y = self.config['source_position'][1] + r * np.sin(theta)
        z = self.config['source_position'][2]
        
        # Initial position and direction
        position = np.array([x, y, z])
        direction = np.array(self.config['source_direction'])
        
        # Create photon packet
        photon = PhotonPacket(position, direction, weight=1.0, wavelength=self.wavelength)
        
        return photon
    
    def get_step_size(self, mu_t):
        """
        Sample the step size from an exponential distribution.
        
        Args:
            mu_t (float): Total interaction coefficient (μₐ + μₛ).
            
        Returns:
            float: Step size.
        """
        # Generate random step size based on exponential distribution
        xi = np.random.random()
        step = -np.log(xi) / mu_t
        
        # Limit step size for numerical stability
        if step > self.config['max_step_size']:
            step = self.config['max_step_size']
        
        return step
    
    def detect_photon(self, photon):
        """
        Check if photon is detected and record it.
        
        Args:
            photon (PhotonPacket): Photon packet to check.
            
        Returns:
            bool: True if photon was detected, False otherwise.
        """
        # Photon must exit at z=0 (surface) to be detected
        if photon.position[2] > 1e-6:
            return False
        
        # Check if photon is within detector area
        detector_x = self.config['detector_position'][0]
        detector_y = self.config['detector_position'][1]
        
        distance = np.sqrt((photon.position[0] - detector_x)**2 + 
                         (photon.position[1] - detector_y)**2)
        
        if distance > self.config['detector_radius']:
            return False
        
        # Check if photon direction is within detection angle
        if photon.direction[2] < np.cos(self.config['detection_angle']):
            return False
        
        # Record detected photon
        ix = np.searchsorted(self.x, photon.position[0]) - 1
        iy = np.searchsorted(self.y, photon.position[1]) - 1
        
        if 0 <= ix < len(self.x)-1 and 0 <= iy < len(self.y)-1:
            self.detection_grid[iy, ix] += photon.weight
        
        # Store detected photon properties
        self.detected_photons.append({
            'position': photon.position.copy(),
            'direction': photon.direction.copy(),
            'weight': photon.weight,
            'path_length': photon.total_distance
        })
        
        return True
    
    def run_simulation(self):
        """
        Run the Monte Carlo simulation with the specified number of photons.
        """
        print(f"Starting Monte Carlo simulation with {self.config['num_photons']} photons at {self.wavelength} wavelength...")
        start_time = time.time()
        
        num_detected = 0
        
        for i in range(self.config['num_photons']):
            # Launch new photon
            photon = self.launch_photon()
            path = []
            
            # Propagate photon until it's either detected, absorbed, or leaves the domain
            while photon.alive:
                # Get optical properties at current position
                mu_a, mu_s, g, n1 = self.get_optical_properties(photon.position)
                mu_t = mu_a + mu_s
                
                # Get step size
                photon.step_size = self.get_step_size(mu_t)
                
                # Move photon
                old_position = photon.position.copy()
                photon.move()
                
                # Check if photon leaves the domain or enters a new tissue type
                new_tissue_type = self.get_tissue_type(photon.position)
                
                if new_tissue_type == 0:
                    # Photon reached boundary - check if detected
                    if photon.position[2] < 1e-6 and photon.direction[2] < 0:
                        # Exiting the top surface
                        if self.detect_photon(photon):
                            num_detected += 1
                    
                    # Terminate photon
                    photon.alive = False
                    continue
                
                # Record absorption
                if mu_a > 0:
                    # Calculate absorption
                    absorption = photon.weight * (mu_a / mu_t) * (1 - np.exp(-mu_t * photon.step_size))
                    
                    # Record absorption in 3D map
                    ix = np.searchsorted(self.x, old_position[0]) - 1
                    iy = np.searchsorted(self.y, old_position[1]) - 1
                    iz = np.searchsorted(self.z, old_position[2]) - 1
                    
                    if (0 <= ix < len(self.x)-1 and 
                        0 <= iy < len(self.y)-1 and 
                        0 <= iz < len(self.z)-1):
                        self.absorption_map[ix, iy, iz] += absorption
                    
                    # Update photon weight
                    photon.weight -= absorption
                
                # Check if photon weight is below threshold
                if photon.weight < self.config['max_weight']:
                    # Russian roulette
                    if np.random.random() < 0.1:
                        photon.weight *= 10
                    else:
                        photon.alive = False
                        continue
                
                # Scatter photon
                if mu_s > 0:
                    photon.scatter(g)
                
                # Record path if requested
                if self.config['record_paths']:
                    path.append(photon.position.copy())
            
            # Progress update
            if (i + 1) % 1000 == 0:
                elapsed_time = time.time() - start_time
                photons_per_second = (i + 1) / elapsed_time
                print(f"Processed {i + 1} photons. "
                     f"Detection rate: {num_detected / (i + 1):.1%}. "
                     f"Speed: {photons_per_second:.1f} photons/second.")
        
        elapsed_time = time.time() - start_time
        detection_rate = num_detected / self.config['num_photons']
        
        print(f"Simulation completed in {elapsed_time:.2f} seconds.")
        print(f"Detected {num_detected} photons ({detection_rate:.2%}).")
        
        return {
            'detection_grid': self.detection_grid,
            'detected_photons': self.detected_photons,
            'absorption_map': self.absorption_map,
            'detection_rate': detection_rate,
            'elapsed_time': elapsed_time
        }
    
    def visualize_results(self, save_dir=None):
        """
        Visualize the simulation results.
        
        Args:
            save_dir (str, optional): Directory to save visualizations. If None, figures will be displayed.
        """
        # Create save directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # 1. Detection grid
        plt.figure(figsize=(10, 8))
        plt.imshow(self.detection_grid, extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],
                  origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Detected Photon Weight')
        plt.xlabel('X Position (cm)')
        plt.ylabel('Y Position (cm)')
        plt.title(f'Detected Photon Distribution ({self.wavelength} wavelength)')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'detection_grid_{self.wavelength}.png'), dpi=300)
            plt.close()
        else:
            plt.show()
        
        # 2. Absorption maps (cross-sections)
        # X-Z cross-section
        plt.figure(figsize=(12, 8))
        mid_y = len(self.y) // 2
        plt.imshow(self.absorption_map[:, mid_y, :].T, 
                  extent=[self.x[0], self.x[-1], self.z[0], self.z[-1]],
                  origin='lower', cmap='hot', interpolation='nearest', aspect='auto')
        plt.colorbar(label='Absorption')
        plt.xlabel('X Position (cm)')
        plt.ylabel('Z Position (cm)')
        plt.title(f'Absorption Map - X-Z Cross-section ({self.wavelength} wavelength)')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'absorption_map_xz_{self.wavelength}.png'), dpi=300)
            plt.close()
        else:
            plt.show()
        
        # Y-Z cross-section
        plt.figure(figsize=(10, 8))
        mid_x = len(self.x) // 2
        plt.imshow(self.absorption_map[mid_x, :, :].T,
                  extent=[self.y[0], self.y[-1], self.z[0], self.z[-1]],
                  origin='lower', cmap='hot', interpolation='nearest')
        plt.colorbar(label='Absorption')
        plt.xlabel('Y Position (cm)')
        plt.ylabel('Z Position (cm)')
        plt.title(f'Absorption Map - Y-Z Cross-section ({self.wavelength} wavelength)')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'absorption_map_yz_{self.wavelength}.png'), dpi=300)
            plt.close()
        else:
            plt.show()
        
        # 3. Path length distribution for detected photons
        if self.detected_photons:
            path_lengths = [photon['path_length'] for photon in self.detected_photons]
            
            plt.figure(figsize=(10, 6))
            plt.hist(path_lengths, bins=50, alpha=0.7, color='blue')
            plt.xlabel('Path Length (cm)')
            plt.ylabel('Number of Photons')
            plt.title(f'Path Length Distribution ({self.wavelength} wavelength)')
            plt.grid(True, alpha=0.3)
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'path_length_dist_{self.wavelength}.png'), dpi=300)
                plt.close()
            else:
                plt.show()
    
    def export_results(self, output_file):
        """
        Export simulation results to a file for later use.
        
        Args:
            output_file (str): Path to the output file.
        """
        try:
            import h5py
            
            with h5py.File(output_file, 'w') as f:
                # Create groups
                optical_group = f.create_group('optical_results')
                
                # Store basic information
                optical_group.attrs['wavelength'] = self.wavelength
                optical_group.attrs['num_photons'] = self.config['num_photons']
                optical_group.attrs['detection_rate'] = len(self.detected_photons) / self.config['num_photons']
                
                # Store grid coordinates
                optical_group.create_dataset('x_grid', data=self.x)
                optical_group.create_dataset('y_grid', data=self.y)
                optical_group.create_dataset('z_grid', data=self.z)
                
                # Store detection grid
                optical_group.create_dataset('detection_grid', data=self.detection_grid)
                
                # Store absorption map (downsampled if needed to save space)
                optical_group.create_dataset('absorption_map', data=self.absorption_map)
                
                # Store detected photon data
                if self.detected_photons:
                    detected_group = optical_group.create_group('detected_photons')
                    
                    positions = np.array([photon['position'] for photon in self.detected_photons])
                    directions = np.array([photon['direction'] for photon in self.detected_photons])
                    weights = np.array([photon['weight'] for photon in self.detected_photons])
                    path_lengths = np.array([photon['path_length'] for photon in self.detected_photons])
                    
                    detected_group.create_dataset('positions', data=positions)
                    detected_group.create_dataset('directions', data=directions)
                    detected_group.create_dataset('weights', data=weights)
                    detected_group.create_dataset('path_lengths', data=path_lengths)
                
                # Store simulation parameters
                params = optical_group.create_group('parameters')
                for key, value in self.config.items():
                    if isinstance(value, (int, float, str, bool)):
                        params.attrs[key] = value
            
            print(f"Optical simulation results exported to {output_file}")
        except ImportError:
            print("h5py not installed. Cannot export results.")


def run_simulation(tissue_model, config=None, output_dir=None):
    """
    Run a complete Monte Carlo simulation with the given configuration.
    
    Args:
        tissue_model (dict): 3D tissue model containing tissue types.
        config (dict, optional): Configuration for the simulation.
        output_dir (str, optional): Directory to save output files.
        
    Returns:
        MonteCarloSimulation: The simulation object with results.
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create and run the simulation
    simulation = MonteCarloSimulation(tissue_model, config)
    simulation.run_simulation()
    
    # Visualize the results
    if output_dir:
        simulation.visualize_results(os.path.join(output_dir, 'figures'))
        
        # Export results
        output_file = os.path.join(output_dir, f'optical_results_{simulation.wavelength}.h5')
        simulation.export_results(output_file)
    
    return simulation


if __name__ == "__main__":
    # Test with a simple tissue model
    from utils.geometry import create_wrist_mesh
    
    # Create a wrist model
    tissue_model = create_wrist_mesh()
    
    # Define simulation configuration
    config = {
        'num_photons': 10000,  # Use a smaller number for testing
        'wavelength': 'red',
        'source_position': [2.0, 0.0, 0.0],  # cm
        'source_direction': [0, 0, 1],
        'source_radius': 0.1,  # cm
        'detector_radius': 0.3,  # cm
        'detector_position': [2.5, 0.0, 0.0],  # cm
        'max_weight': 1e-4,
        'record_paths': True
    }
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             'data', 'results', 'optical')
    
    # Run the simulation
    simulation = run_simulation(tissue_model, config, output_dir)
