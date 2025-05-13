#!/usr/bin/env python3
"""
Utility functions for creating anatomical models of the wrist for simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from .constants import (RADIAL_ARTERY_RADIUS, RADIAL_ARTERY_DEPTH,
                       ULNAR_ARTERY_RADIUS, ULNAR_ARTERY_DEPTH,
                       EPIDERMIS_THICKNESS, DERMIS_THICKNESS,
                       SUBCUTANEOUS_FAT_THICKNESS)


def create_arterial_model(config=None):
    """
    Create a simplified 3D model of the radial artery in the wrist.
    
    Args:
        config (dict, optional): Configuration parameters for the arterial model.
            If None, default parameters are used.
    
    Returns:
        dict: Dictionary containing the arterial model data.
    """
    # Default configuration
    if config is None:
        config = {
            'artery_radius': RADIAL_ARTERY_RADIUS,
            'artery_depth': RADIAL_ARTERY_DEPTH,
            'artery_length': 5.0,  # cm
            'curvature': 0.1,      # Curvature factor
            'bifurcation': True,   # Whether to include bifurcation
            'grid_resolution': 0.05,  # cm
        }
    
    # Extract parameters
    radius = config['artery_radius']
    depth = config['artery_depth']
    length = config['artery_length']
    curvature = config['curvature']
    include_bifurcation = config['bifurcation']
    resolution = config['grid_resolution']
    
    # Create coordinate grids
    x = np.arange(0, length + resolution, resolution)
    y = np.arange(-2*radius, 2*radius + resolution, resolution)
    z = np.arange(0, depth + 3*radius + resolution, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create the main artery using a cylinder with slight curvature
    # Centerline of the artery
    centerline_y = curvature * np.sin(np.pi * x / length)
    centerline_z = depth + radius + 0.5 * curvature * (1 - np.cos(np.pi * x / length))
    
    # Initialize volume fraction array (0 = tissue, 1 = blood)
    volume_fraction = np.zeros_like(X)
    
    # Fill the main artery
    for i, x_i in enumerate(x):
        distance_from_centerline = np.sqrt((Y[i,:,:] - centerline_y[i])**2 + 
                                         (Z[i,:,:] - centerline_z[i])**2)
        volume_fraction[i,:,:] = (distance_from_centerline <= radius).astype(float)
    
    # Create bifurcation if requested
    if include_bifurcation:
        # Bifurcation parameters
        bifurcation_start = int(0.7 * len(x))
        bifurcation_angle = 30  # degrees
        
        # Convert to radians
        bifurcation_angle_rad = bifurcation_angle * np.pi / 180
        
        # Length of bifurcation
        bifurcation_length = length - x[bifurcation_start]
        
        # Points along the bifurcation
        x_bifurcation = x[bifurcation_start:]
        
        # Centerlines for the two branches
        branch1_y = centerline_y[bifurcation_start] + (x_bifurcation - x[bifurcation_start]) * np.sin(bifurcation_angle_rad)
        branch1_z = centerline_z[bifurcation_start] + 0.5 * (1 - np.cos(bifurcation_angle_rad)) * (x_bifurcation - x[bifurcation_start])
        
        branch2_y = centerline_y[bifurcation_start] - (x_bifurcation - x[bifurcation_start]) * np.sin(bifurcation_angle_rad)
        branch2_z = centerline_z[bifurcation_start] + 0.5 * (1 - np.cos(bifurcation_angle_rad)) * (x_bifurcation - x[bifurcation_start])
        
        # Smaller radius for branches
        branch_radius = 0.7 * radius
        
        # Fill the branches
        for i, x_i in enumerate(x_bifurcation, bifurcation_start):
            # Branch 1
            distance_from_branch1 = np.sqrt((Y[i,:,:] - branch1_y[i-bifurcation_start])**2 + 
                                          (Z[i,:,:] - branch1_z[i-bifurcation_start])**2)
            branch1_volume = (distance_from_branch1 <= branch_radius).astype(float)
            
            # Branch 2
            distance_from_branch2 = np.sqrt((Y[i,:,:] - branch2_y[i-bifurcation_start])**2 + 
                                          (Z[i,:,:] - branch2_z[i-bifurcation_start])**2)
            branch2_volume = (distance_from_branch2 <= branch_radius).astype(float)
            
            # Combine branches
            volume_fraction[i,:,:] = np.maximum(branch1_volume, branch2_volume)
    
    # Create the tissue layers
    tissue_types = np.zeros_like(X, dtype=int)  # 0=outside, 1=epidermis, 2=dermis, 3=fat, 4=muscle
    
    # Surface at z=0
    tissue_types[:,:,0] = 1  # Epidermis at surface
    
    # Epidermis layer
    epidermis_depth = int(EPIDERMIS_THICKNESS / resolution)
    tissue_types[:,:,1:epidermis_depth+1] = 1
    
    # Dermis layer
    dermis_depth = int(DERMIS_THICKNESS / resolution)
    tissue_types[:,:,epidermis_depth+1:epidermis_depth+dermis_depth+1] = 2
    
    # Fat layer
    fat_depth = int(SUBCUTANEOUS_FAT_THICKNESS / resolution)
    tissue_types[:,:,epidermis_depth+dermis_depth+1:epidermis_depth+dermis_depth+fat_depth+1] = 3
    
    # Muscle layer (everything deeper)
    tissue_types[:,:,epidermis_depth+dermis_depth+fat_depth+1:] = 4
    
    # Blood overwrites tissue where volume_fraction is 1
    blood_mask = volume_fraction > 0.5
    tissue_types[blood_mask] = 5  # 5 = blood
    
    # Compile the model data
    model = {
        'x': x,
        'y': y,
        'z': z,
        'X': X,
        'Y': Y,
        'Z': Z,
        'volume_fraction': volume_fraction,
        'tissue_types': tissue_types,
        'centerline_y': centerline_y,
        'centerline_z': centerline_z,
        'config': config
    }
    
    if include_bifurcation:
        model.update({
            'bifurcation_start': bifurcation_start,
            'branch1_y': branch1_y,
            'branch1_z': branch1_z,
            'branch2_y': branch2_y,
            'branch2_z': branch2_z
        })
    
    return model


def visualize_arterial_model(model, save_path=None):
    """
    Visualize the 3D arterial model.
    
    Args:
        model (dict): The arterial model created with create_arterial_model.
        save_path (str, optional): Path to save the visualization. If None, the plot is displayed.
    """
    # Extract data from the model
    X, Y, Z = model['X'], model['Y'], model['Z']
    volume_fraction = model['volume_fraction']
    tissue_types = model['tissue_types']
    
    # Create a color map for tissue types
    tissue_colors = {
        0: 'white',     # outside
        1: 'pink',      # epidermis
        2: 'lightcoral', # dermis
        3: 'yellow',    # fat
        4: 'brown',     # muscle
        5: 'red'        # blood
    }
    
    # 2D cross-section at the middle of the model
    mid_x = X.shape[0] // 2
    
    plt.figure(figsize=(12, 8))
    
    # Plot tissue types
    plt.subplot(1, 2, 1)
    plt.title('Tissue Types (Cross-section)')
    tissue_cross = tissue_types[mid_x, :, :]
    
    # Create a custom colormap
    from matplotlib.colors import ListedColormap
    colors = [tissue_colors[i] for i in range(6)]
    cmap = ListedColormap(colors)
    
    plt.imshow(tissue_cross.T, origin='lower', cmap=cmap, interpolation='none',
               extent=[model['y'][0], model['y'][-1], model['z'][0], model['z'][-1]])
    plt.colorbar(ticks=np.arange(6), label='Tissue Type')
    plt.xlabel('Y Position (cm)')
    plt.ylabel('Z Position (cm)')
    
    # Plot blood volume fraction
    plt.subplot(1, 2, 2)
    plt.title('Blood Volume Fraction (Cross-section)')
    blood_cross = volume_fraction[mid_x, :, :]
    plt.imshow(blood_cross.T, origin='lower', cmap='Reds', interpolation='none',
               extent=[model['y'][0], model['y'][-1], model['z'][0], model['z'][-1]])
    plt.colorbar(label='Blood Volume Fraction')
    plt.xlabel('Y Position (cm)')
    plt.ylabel('Z Position (cm)')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
    
    # 3D visualization of the artery
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the centerline
    x = model['x']
    centerline_y = model['centerline_y']
    centerline_z = model['centerline_z']
    
    ax.plot(x, centerline_y, centerline_z, 'r-', linewidth=2, label='Artery Centerline')
    
    # Plot bifurcation if present
    if 'bifurcation_start' in model:
        bifurcation_start = model['bifurcation_start']
        branch1_y = model['branch1_y']
        branch1_z = model['branch1_z']
        branch2_y = model['branch2_y']
        branch2_z = model['branch2_z']
        
        x_bifurcation = x[bifurcation_start:]
        
        ax.plot(x_bifurcation, branch1_y, branch1_z, 'b-', linewidth=2, label='Branch 1')
        ax.plot(x_bifurcation, branch2_y, branch2_z, 'g-', linewidth=2, label='Branch 2')
    
    ax.set_xlabel('X Position (cm)')
    ax.set_ylabel('Y Position (cm)')
    ax.set_zlabel('Z Position (cm)')
    ax.set_title('3D Arterial Model')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path.replace('.png', '_3d.png'), dpi=300)
        plt.close()
    else:
        plt.show()


def create_wrist_mesh(config=None):
    """
    Create a more detailed 3D mesh model of the wrist for optical simulations.
    
    Args:
        config (dict, optional): Configuration parameters.
    
    Returns:
        dict: Dictionary containing the wrist mesh data.
    """
    # Default configuration
    if config is None:
        config = {
            'width': 5.0,         # cm (wrist width)
            'height': 2.5,        # cm (wrist height)
            'length': 4.0,        # cm (segment length)
            'grid_resolution': 0.05,  # cm
            'skin_thickness': EPIDERMIS_THICKNESS + DERMIS_THICKNESS,
            'fat_thickness': SUBCUTANEOUS_FAT_THICKNESS,
            'include_vessels': True,
            'include_bone': True
        }
    
    # Extract parameters
    width = config['width']
    height = config['height']
    length = config['length']
    resolution = config['grid_resolution']
    
    # Create coordinate grids
    x = np.arange(0, length + resolution, resolution)
    y = np.arange(-width/2, width/2 + resolution, resolution)
    z = np.arange(0, height + resolution, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Initialize tissue types array
    # 0=outside, 1=epidermis, 2=dermis, 3=fat, 4=muscle, 5=blood, 6=bone
    tissue_types = np.zeros_like(X, dtype=int)
    
    # Create wrist shape (elliptical cylinder)
    wrist_shape = ((Y / (width/2))**2 + (Z / height)**2) <= 1.0
    
    # Fill in basic tissue layers
    # First define tissue depths
    epidermis_depth = EPIDERMIS_THICKNESS
    dermis_depth = DERMIS_THICKNESS
    skin_depth = epidermis_depth + dermis_depth
    fat_depth = SUBCUTANEOUS_FAT_THICKNESS
    
    # Define the wrist surface
    surface_dist = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Find the distance from the outside of the wrist
            if not np.any(wrist_shape[i, j, :]):
                # This y,x position is entirely outside the wrist
                continue
                
            # Find first point inside the wrist
            first_inside = np.argmax(wrist_shape[i, j, :])
            surface_dist[i, j, :] = np.abs(np.arange(Z.shape[2]) - first_inside) * resolution
    
    # Create tissue layers based on distance from surface
    epidermis_mask = (surface_dist <= epidermis_depth) & wrist_shape
    dermis_mask = (surface_dist > epidermis_depth) & (surface_dist <= skin_depth) & wrist_shape
    fat_mask = (surface_dist > skin_depth) & (surface_dist <= skin_depth + fat_depth) & wrist_shape
    
    tissue_types[epidermis_mask] = 1  # Epidermis
    tissue_types[dermis_mask] = 2     # Dermis
    tissue_types[fat_mask] = 3        # Fat
    
    # All remaining wrist volume is muscle by default
    muscle_mask = wrist_shape & (tissue_types == 0)
    tissue_types[muscle_mask] = 4     # Muscle
    
    # Add blood vessels if requested
    if config['include_vessels']:
        # Radial artery
        radial_y = -width/4
        radial_z = skin_depth + RADIAL_ARTERY_DEPTH
        radial_radius = RADIAL_ARTERY_RADIUS
        
        # Calculate distance from radial artery centerline
        radial_dist = np.sqrt((Y - radial_y)**2 + (Z - radial_z)**2)
        radial_mask = (radial_dist <= radial_radius) & wrist_shape
        
        # Ulnar artery
        ulnar_y = width/4
        ulnar_z = skin_depth + ULNAR_ARTERY_DEPTH
        ulnar_radius = ULNAR_ARTERY_RADIUS
        
        # Calculate distance from ulnar artery centerline
        ulnar_dist = np.sqrt((Y - ulnar_y)**2 + (Z - ulnar_z)**2)
        ulnar_mask = (ulnar_dist <= ulnar_radius) & wrist_shape
        
        # Set blood vessels
        tissue_types[radial_mask | ulnar_mask] = 5  # Blood
    
    # Add bone if requested
    if config['include_bone']:
        # Simplified bone structure - two parallel bones
        radius_y = -width/6
        radius_z = height/2
        radius_radius = 0.4  # cm
        
        ulna_y = width/6
        ulna_z = height/2
        ulna_radius = 0.35  # cm
        
        # Calculate distances
        radius_dist = np.sqrt((Y - radius_y)**2 + (Z - radius_z)**2)
        ulna_dist = np.sqrt((Y - ulna_y)**2 + (Z - ulna_z)**2)
        
        # Create bone masks
        radius_mask = (radius_dist <= radius_radius) & wrist_shape
        ulna_mask = (ulna_dist <= ulna_radius) & wrist_shape
        
        # Set bones
        tissue_types[radius_mask | ulna_mask] = 6  # Bone
    
    # Compile the model data
    wrist_model = {
        'x': x,
        'y': y,
        'z': z,
        'X': X,
        'Y': Y,
        'Z': Z,
        'tissue_types': tissue_types,
        'config': config
    }
    
    if config['include_vessels']:
        wrist_model.update({
            'radial_y': radial_y,
            'radial_z': radial_z,
            'radial_radius': radial_radius,
            'ulnar_y': ulnar_y,
            'ulnar_z': ulnar_z,
            'ulnar_radius': ulnar_radius
        })
    
    return wrist_model


def visualize_wrist_model(model, save_path=None):
    """
    Visualize the 3D wrist model.
    
    Args:
        model (dict): The wrist model created with create_wrist_mesh.
        save_path (str, optional): Path to save the visualization. If None, the plot is displayed.
    """
    # Extract data from the model
    tissue_types = model['tissue_types']
    
    # Create a color map for tissue types
    tissue_colors = {
        0: 'white',      # outside
        1: 'pink',       # epidermis
        2: 'lightcoral', # dermis
        3: 'yellow',     # fat
        4: 'brown',      # muscle
        5: 'red',        # blood
        6: 'lightgray'   # bone
    }
    
    # Get cross-sections at different positions
    mid_x = tissue_types.shape[0] // 2
    mid_y = tissue_types.shape[1] // 2
    mid_z = tissue_types.shape[2] // 2
    
    # Create a custom colormap
    from matplotlib.colors import ListedColormap
    colors = [tissue_colors[i] for i in range(7)]
    cmap = ListedColormap(colors)
    
    plt.figure(figsize=(18, 6))
    
    # Transverse cross-section (X-Y plane)
    plt.subplot(1, 3, 1)
    plt.title('Transverse Cross-section (X-Y plane)')
    transverse = tissue_types[:, :, mid_z]
    plt.imshow(transverse.T, origin='lower', cmap=cmap, interpolation='none',
               extent=[model['x'][0], model['x'][-1], model['y'][0], model['y'][-1]])
    plt.colorbar(ticks=np.arange(7), label='Tissue Type')
    plt.xlabel('X Position (cm)')
    plt.ylabel('Y Position (cm)')
    
    # Coronal cross-section (X-Z plane)
    plt.subplot(1, 3, 2)
    plt.title('Coronal Cross-section (X-Z plane)')
    coronal = tissue_types[:, mid_y, :]
    plt.imshow(coronal.T, origin='lower', cmap=cmap, interpolation='none',
               extent=[model['x'][0], model['x'][-1], model['z'][0], model['z'][-1]])
    plt.colorbar(ticks=np.arange(7), label='Tissue Type')
    plt.xlabel('X Position (cm)')
    plt.ylabel('Z Position (cm)')
    
    # Sagittal cross-section (Y-Z plane)
    plt.subplot(1, 3, 3)
    plt.title('Sagittal Cross-section (Y-Z plane)')
    sagittal = tissue_types[mid_x, :, :]
    plt.imshow(sagittal.T, origin='lower', cmap=cmap, interpolation='none',
               extent=[model['y'][0], model['y'][-1], model['z'][0], model['z'][-1]])
    plt.colorbar(ticks=np.arange(7), label='Tissue Type')
    plt.xlabel('Y Position (cm)')
    plt.ylabel('Z Position (cm)')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Test creating and visualizing the arterial model
    model = create_arterial_model()
    visualize_arterial_model(model)
    
    # Test creating and visualizing the wrist model
    wrist_model = create_wrist_mesh()
    visualize_wrist_model(wrist_model)
