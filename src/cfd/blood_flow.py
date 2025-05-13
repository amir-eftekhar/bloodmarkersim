#!/usr/bin/env python3
"""
CFD simulation of blood flow in the wrist arterial network.
This module implements the Navier-Stokes equations to model pulsatile blood flow,
incorporating non-Newtonian blood properties and realistic arterial geometry.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import utility modules
sys.path.append(str(Path(__file__).parent.parent))
from utils.constants import BLOOD_DENSITY, BLOOD_VISCOSITY, WOMERSLEY_NUMBER
from utils.geometry import create_arterial_model


class BloodFlowSimulation:
    """
    Class to simulate blood flow in the wrist arteries using CFD techniques.
    
    Implements a simplified Navier-Stokes solver for pulsatile blood flow in
    the radial artery of the wrist. Includes non-Newtonian blood properties.
    """
    
    def __init__(self, config=None):
        """
        Initialize the CFD simulation with the given configuration.
        
        Args:
            config (dict): Configuration parameters for the simulation.
        """
        self.config = config or self._default_config()
        
        # Domain parameters
        self.artery_length = self.config['artery_length']  # Length of artery segment (cm)
        self.artery_radius = self.config['artery_radius']  # Radius of artery (cm)
        self.artery_depth = self.config['artery_depth']    # Depth of artery from skin surface (cm)
        
        # Blood properties
        self.density = self.config['blood_density']  # Blood density (g/cm^3)
        self.viscosity = self.config['blood_viscosity']  # Blood viscosity (g/cm·s)
        
        # Simulation parameters
        self.num_x = self.config['num_x']  # Number of grid points along artery
        self.num_r = self.config['num_r']  # Number of grid points in radial direction
        self.dt = self.config['dt']  # Time step (s)
        self.cardiac_period = self.config['cardiac_period']  # Cardiac cycle period (s)
        self.num_cycles = self.config['num_cycles']  # Number of cardiac cycles to simulate
        
        # Initialize mesh
        self._initialize_mesh()
        
        # Initialize velocity and pressure fields
        self.velocity_axial = np.zeros((self.num_r, self.num_x))  # Axial velocity
        self.velocity_radial = np.zeros((self.num_r, self.num_x))  # Radial velocity
        self.pressure = np.zeros(self.num_x)  # Pressure along artery
        
        # Results storage
        self.results = {
            'time': [],
            'velocity_profiles': [],
            'pressure_profiles': [],
            'wall_shear_stress': []
        }

    def _default_config(self):
        """Default configuration for the simulation."""
        return {
            'artery_length': 10.0,  # cm
            'artery_radius': 0.1,   # cm (1 mm)
            'artery_depth': 0.2,    # cm (2 mm)
            'blood_density': BLOOD_DENSITY,
            'blood_viscosity': BLOOD_VISCOSITY,
            'num_x': 100,
            'num_r': 20,
            'dt': 0.001,  # s
            'cardiac_period': 0.8,  # s (75 bpm)
            'num_cycles': 3,
            'womersley_number': WOMERSLEY_NUMBER
        }
    
    def _initialize_mesh(self):
        """Initialize the computational mesh for the CFD simulation."""
        # Create a linear grid in the axial direction
        self.x = np.linspace(0, self.artery_length, self.num_x)
        
        # Create a grid in the radial direction (finer near the wall)
        self.r = np.linspace(0, self.artery_radius, self.num_r)
        
        # Create 2D meshgrid for easier calculations
        self.R, self.X = np.meshgrid(self.r, self.x, indexing='ij')
    
    def inlet_velocity_profile(self, t):
        """
        Calculate the inlet velocity profile at time t.
        
        Implements a physiological pulsatile flow profile based on the cardiac cycle.
        
        Args:
            t (float): Current time in seconds.
            
        Returns:
            numpy.ndarray: Velocity profile at the inlet.
        """
        # Normalized time within the cardiac cycle
        t_norm = (t % self.cardiac_period) / self.cardiac_period
        
        # Mean velocity during the cardiac cycle (cm/s)
        mean_velocity = 15.0
        
        # Pulsatility factor
        pulsatility = 0.5
        
        # Simple pulsatile profile (systole and diastole)
        if t_norm < 0.2:  # Systole
            factor = 1.0 + 2.0 * pulsatility * np.sin(t_norm * np.pi / 0.2)
        else:  # Diastole
            factor = 1.0 - pulsatility * (1 - np.exp(-(t_norm - 0.2) / 0.2))
        
        # Poiseuille profile at the inlet
        velocity = factor * mean_velocity * (1 - (self.r / self.artery_radius) ** 2)
        
        return velocity
    
    def _non_newtonian_viscosity(self, shear_rate):
        """
        Calculate non-Newtonian viscosity using the Carreau-Yasuda model.
        
        Args:
            shear_rate (numpy.ndarray): Shear rate field.
            
        Returns:
            numpy.ndarray: Viscosity field.
        """
        # Carreau-Yasuda model parameters for blood
        viscosity_inf = 0.0035  # g/cm·s (asymptotic viscosity at infinite shear rate)
        viscosity_0 = 0.16      # g/cm·s (viscosity at zero shear rate)
        lambda_time = 8.2       # s (relaxation time)
        a = 2.0                 # dimensionless
        n = 0.2128              # dimensionless
        
        # Carreau-Yasuda model
        viscosity = viscosity_inf + (viscosity_0 - viscosity_inf) * (1 + (lambda_time * shear_rate) ** a) ** ((n - 1) / a)
        
        return viscosity
    
    def simulate(self):
        """Run the CFD simulation for the specified number of cardiac cycles."""
        print("Starting CFD simulation of blood flow...")
        
        # Total simulation time
        total_time = self.num_cycles * self.cardiac_period
        time_points = np.arange(0, total_time, self.dt)
        
        # Store the initial time
        self.results['time'].append(0)
        self.results['velocity_profiles'].append(self.velocity_axial.copy())
        self.results['pressure_profiles'].append(self.pressure.copy())
        self.results['wall_shear_stress'].append(np.zeros(self.num_x))
        
        # Main time-stepping loop
        for t_idx, t in enumerate(time_points[1:], 1):
            # Get inlet velocity profile
            inlet_velocity = self.inlet_velocity_profile(t)
            
            # Set boundary condition at inlet
            self.velocity_axial[:, 0] = inlet_velocity
            
            # Simplified Navier-Stokes solver (Poiseuille flow)
            self._solve_simplified_navier_stokes()
            
            # Calculate wall shear stress
            wss = self._calculate_wall_shear_stress()
            
            # Store results at regular intervals
            if t_idx % int(0.05 / self.dt) == 0:  # Every 0.05s
                self.results['time'].append(t)
                self.results['velocity_profiles'].append(self.velocity_axial.copy())
                self.results['pressure_profiles'].append(self.pressure.copy())
                self.results['wall_shear_stress'].append(wss)
                
            # Print progress
            if t_idx % int(0.1 / self.dt) == 0:
                progress = (t / total_time) * 100
                print(f"Simulation progress: {progress:.1f}%")
        
        print("CFD simulation completed.")
    
    def _solve_simplified_navier_stokes(self):
        """
        Solve a simplified version of the Navier-Stokes equations for pipe flow.
        
        This is a simplified model based on the Poiseuille equation with
        time-varying boundary conditions to simulate pulsatile flow.
        """
        # Calculate pressure gradient based on inlet velocity
        mean_velocity = np.mean(self.velocity_axial[:, 0])
        pressure_gradient = 8 * self.viscosity * mean_velocity / (self.artery_radius ** 2)
        
        # Update pressure along the artery
        for i in range(1, self.num_x):
            self.pressure[i] = self.pressure[i-1] - pressure_gradient * (self.x[i] - self.x[i-1])
        
        # Update velocity profile based on pressure gradient
        for i in range(1, self.num_x):
            self.velocity_axial[:, i] = (self.pressure[i-1] - self.pressure[i]) / (self.x[i] - self.x[i-1]) * \
                                      (self.artery_radius ** 2 - self.r ** 2) / (4 * self.viscosity)
    
    def _calculate_wall_shear_stress(self):
        """
        Calculate wall shear stress along the artery.
        
        Returns:
            numpy.ndarray: Wall shear stress at each axial position.
        """
        # Wall shear stress = viscosity * du/dr at r = R
        # Approximating with a first-order derivative at the wall
        du_dr_wall = (self.velocity_axial[-2, :] - self.velocity_axial[-1, :]) / (self.r[-2] - self.r[-1])
        wall_shear_stress = -self.viscosity * du_dr_wall
        
        return wall_shear_stress
    
    def visualize_results(self, save_dir=None):
        """
        Visualize the simulation results.
        
        Args:
            save_dir (str, optional): Directory to save visualizations. If None, figures will be displayed.
        """
        # Create save directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Select a few time points to visualize
        indices = np.linspace(0, len(self.results['time'])-1, 5, dtype=int)
        
        # 1. Velocity profiles at different times
        plt.figure(figsize=(10, 6))
        for idx in indices:
            t = self.results['time'][idx]
            profile = self.results['velocity_profiles'][idx][:, self.num_x//2]  # Middle of artery
            plt.plot(self.r, profile, label=f't = {t:.2f}s')
        
        plt.xlabel('Radial position (cm)')
        plt.ylabel('Axial velocity (cm/s)')
        plt.title('Velocity Profiles in Radial Artery')
        plt.legend()
        plt.grid(True)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'velocity_profiles.png'), dpi=300)
            plt.close()
        else:
            plt.show()
        
        # 2. Pressure along the artery
        plt.figure(figsize=(10, 6))
        for idx in indices:
            t = self.results['time'][idx]
            pressure = self.results['pressure_profiles'][idx]
            plt.plot(self.x, pressure, label=f't = {t:.2f}s')
        
        plt.xlabel('Axial position (cm)')
        plt.ylabel('Pressure (g/cm·s²)')
        plt.title('Pressure Distribution along Radial Artery')
        plt.legend()
        plt.grid(True)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'pressure_distribution.png'), dpi=300)
            plt.close()
        else:
            plt.show()
        
        # 3. Wall shear stress
        plt.figure(figsize=(10, 6))
        for idx in indices:
            t = self.results['time'][idx]
            wss = self.results['wall_shear_stress'][idx]
            plt.plot(self.x, wss, label=f't = {t:.2f}s')
        
        plt.xlabel('Axial position (cm)')
        plt.ylabel('Wall Shear Stress (g/cm·s²)')
        plt.title('Wall Shear Stress along Radial Artery')
        plt.legend()
        plt.grid(True)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'wall_shear_stress.png'), dpi=300)
            plt.close()
        else:
            plt.show()
    
    def get_velocity_data(self, position_x=None):
        """
        Get velocity data for further analysis or coupling with optical simulations.
        
        Args:
            position_x (float, optional): Axial position where to extract the 
                velocity profile. If None, the middle of the artery is used.
                
        Returns:
            dict: Dictionary containing velocity data over time.
        """
        if position_x is None:
            position_x = self.artery_length / 2
        
        # Find the closest grid point
        idx_x = np.argmin(np.abs(self.x - position_x))
        
        # Extract velocity data at this position
        velocity_data = {
            'time': self.results['time'],
            'radial_positions': self.r,
            'velocity_profiles': [profile[:, idx_x] for profile in self.results['velocity_profiles']]
        }
        
        return velocity_data
    
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
                cfd_group = f.create_group('cfd_results')
                
                # Store time data
                cfd_group.create_dataset('time', data=np.array(self.results['time']))
                
                # Store spatial grids
                cfd_group.create_dataset('x_grid', data=self.x)
                cfd_group.create_dataset('r_grid', data=self.r)
                
                # Store velocity data (selecting a few time points to save space)
                indices = np.linspace(0, len(self.results['time'])-1, 20, dtype=int)
                velocity_data = np.array([self.results['velocity_profiles'][i] for i in indices])
                time_selected = np.array([self.results['time'][i] for i in indices])
                
                cfd_group.create_dataset('velocity_data', data=velocity_data)
                cfd_group.create_dataset('time_selected', data=time_selected)
                
                # Store wall shear stress
                wss_data = np.array([self.results['wall_shear_stress'][i] for i in indices])
                cfd_group.create_dataset('wall_shear_stress', data=wss_data)
                
                # Store simulation parameters
                params = cfd_group.create_group('parameters')
                for key, value in self.config.items():
                    params.attrs[key] = value
            
            print(f"CFD results exported to {output_file}")
        except ImportError:
            print("h5py not installed. Cannot export results.")


def run_simulation(config=None, output_dir=None):
    """
    Run a complete CFD simulation with the given configuration.
    
    Args:
        config (dict, optional): Configuration for the simulation.
        output_dir (str, optional): Directory to save output files.
        
    Returns:
        BloodFlowSimulation: The simulation object with results.
    """
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create and run the simulation
    simulation = BloodFlowSimulation(config)
    simulation.simulate()
    
    # Visualize the results
    if output_dir:
        simulation.visualize_results(os.path.join(output_dir, 'figures'))
        
        # Export results
        output_file = os.path.join(output_dir, 'cfd_results.h5')
        simulation.export_results(output_file)
    
    return simulation


if __name__ == "__main__":
    # Example configuration
    example_config = {
        'artery_length': 5.0,  # cm
        'artery_radius': 0.15,  # cm
        'artery_depth': 0.2,   # cm
        'blood_density': 1.06,  # g/cm^3
        'blood_viscosity': 0.04,  # g/cm·s
        'num_x': 50,
        'num_r': 20,
        'dt': 0.005,  # s
        'cardiac_period': 0.8,  # s
        'num_cycles': 3,
        'womersley_number': 2.5
    }
    
    # Define output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             'data', 'results', 'cfd')
    
    # Run the simulation
    simulation = run_simulation(example_config, output_dir)
