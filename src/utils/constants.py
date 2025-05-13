#!/usr/bin/env python3
"""
Constants and parameters for the wrist blood flow simulation.
"""

# Blood properties
BLOOD_DENSITY = 1.06  # g/cm^3
BLOOD_VISCOSITY = 0.04  # g/cm·s
WOMERSLEY_NUMBER = 2.5  # dimensionless

# Anatomical parameters
RADIAL_ARTERY_RADIUS = 0.15  # cm (1.5 mm)
RADIAL_ARTERY_DEPTH = 0.2  # cm (2 mm from surface)
ULNAR_ARTERY_RADIUS = 0.14  # cm (1.4 mm)
ULNAR_ARTERY_DEPTH = 0.25  # cm (2.5 mm from surface)

# Tissue thicknesses
EPIDERMIS_THICKNESS = 0.01  # cm (0.1 mm)
DERMIS_THICKNESS = 0.2  # cm (2 mm)
SUBCUTANEOUS_FAT_THICKNESS = 0.3  # cm (3 mm)

# Optical properties - Absorption coefficients (μₐ) in cm⁻¹
# Green wavelength (530-560 nm)
ABSORPTION_GREEN = {
    'oxyhemoglobin': 150.0,
    'deoxyhemoglobin': 200.0,
    'water': 0.1,
    'lipid': 0.3,
    'melanin': 25.0,
    'epidermis': 15.0,
    'dermis': 5.0,
    'fat': 1.0,
    'muscle': 2.5
}

# Red wavelength (630-670 nm)
ABSORPTION_RED = {
    'oxyhemoglobin': 10.0,
    'deoxyhemoglobin': 50.0,
    'water': 0.3,
    'lipid': 0.5,
    'melanin': 15.0,
    'epidermis': 8.0,
    'dermis': 3.0,
    'fat': 0.8,
    'muscle': 2.0
}

# Near-infrared wavelength (850-950 nm)
ABSORPTION_NIR = {
    'oxyhemoglobin': 2.0,
    'deoxyhemoglobin': 3.5,
    'water': 2.0,
    'lipid': 2.5,
    'melanin': 5.0,
    'epidermis': 3.0,
    'dermis': 2.0,
    'fat': 1.2,
    'muscle': 1.8
}

# Optical properties - Scattering coefficients (μₛ) in cm⁻¹
# Green wavelength (530-560 nm)
SCATTERING_GREEN = {
    'blood': 300.0,
    'epidermis': 60.0,
    'dermis': 45.0,
    'fat': 30.0,
    'muscle': 25.0
}

# Red wavelength (630-670 nm)
SCATTERING_RED = {
    'blood': 250.0,
    'epidermis': 50.0,
    'dermis': 40.0,
    'fat': 25.0,
    'muscle': 20.0
}

# Near-infrared wavelength (850-950 nm)
SCATTERING_NIR = {
    'blood': 200.0,
    'epidermis': 40.0,
    'dermis': 30.0,
    'fat': 20.0,
    'muscle': 15.0
}

# Anisotropy factor (g)
ANISOTROPY = {
    'blood': 0.99,
    'epidermis': 0.9,
    'dermis': 0.85,
    'fat': 0.8,
    'muscle': 0.9
}

# Refractive indices
REFRACTIVE_INDEX = {
    'air': 1.0,
    'blood': 1.4,
    'epidermis': 1.35,
    'dermis': 1.4,
    'fat': 1.45,
    'muscle': 1.37
}

# Physiological parameters
HEART_RATE = 75  # beats per minute
CARDIAC_PERIOD = 60 / HEART_RATE  # seconds
OXYGEN_SATURATION = 0.98  # 98% baseline
GLUCOSE_BASELINE = 100  # mg/dL
LIPIDS_BASELINE = {
    'total_cholesterol': 200,  # mg/dL
    'hdl': 50,  # mg/dL
    'ldl': 130,  # mg/dL
    'triglycerides': 150  # mg/dL
}

# Simulation parameters
MONTE_CARLO_PHOTONS = 10000  # Number of photons per simulation
TISSUE_GRID_RESOLUTION = 0.01  # cm
MAX_SIMULATION_DEPTH = 1.0  # cm
