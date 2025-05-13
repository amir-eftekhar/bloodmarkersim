#!/usr/bin/env python3
"""
Physics-Informed Optical Biomarker Prediction Framework (Comprehensive Version)

Combines domain-specific optical physics with advanced deep learning to accurately
predict multiple biomarkers (glucose, SpO2, total_cholesterol, hemoglobin, ldl, hdl, triglycerides)
from multi-wavelength PPG signals, with real-time adaptation capabilities.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew, iqr, median_abs_deviation
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from datetime import datetime
from pathlib import Path
import joblib
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from warnings import warn
import logging
import inspect

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

# TensorFlow and related ML libraries
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Conv1D, Conv2D, Bidirectional, Dropout, BatchNormalization,
    Input, Concatenate, Average, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Reshape, Multiply, Add, TimeDistributed, Flatten,
    LeakyReLU, PReLU, ELU, Activation, MaxPooling1D, AveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras.optimizers import Adam, AdamW, Nadam, RMSprop, SGD # Added SGD
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.constraints import MinMaxNorm, NonNeg, UnitNorm
from tensorflow.keras.initializers import GlorotNormal, GlorotUniform, HeNormal, HeUniform, Orthogonal
from tensorflow.keras.utils import Sequence as KerasSequence # For custom data generators
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

# Scikit-learn and related utilities
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, MinMaxScaler, MaxAbsScaler, QuantileTransformer, KBinsDiscretizer
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GroupKFold, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor, AdaBoostRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer, d2_tweedie_score, mean_pinball_loss
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, RFE, RFECV, SelectKBest, f_regression, SequentialFeatureSelector
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.linear_model import HuberRegressor, Ridge, Lasso, ElasticNet, BayesianRidge, PoissonRegressor, TweedieRegressor, GammaRegressor, OrthogonalMatchingPursuit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.manifold import TSNE, Isomap
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import NotFittedError

# Ray Tune for Hyperparameter Optimization
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, HyperBandScheduler, MedianStoppingRule
    from ray.tune.search import basic_variant_generator
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search.bohb import TuneBOHB
    from ray.tune.integration.keras import TuneReportCallback
    has_ray_tune = True
except ImportError:
    has_ray_tune = False
    warn("Ray Tune not available. Hyperparameter optimization will be limited.")

# PyWavelets for WaveletFeatureExtractor
try:
    import pywt
    has_pywavelets = True
except ImportError:
    has_pywavelets = False
    warn("PyWavelets not available. WaveletFeatureExtractor will be disabled.")

# Nolds for non-linear dynamics features
try:
    import nolds
    has_nolds = True
except ImportError:
    has_nolds = False
    warn("Nolds library not available. PhaseSpaceTransformer advanced features will be limited.")

# HeartPy for PPG processing (optional, for more advanced feature extraction)
try:
    import heartpy as hp
    has_heartpy = True
except ImportError:
    has_heartpy = False
    # warn("HeartPy not available. Some advanced PPG feature extraction might be limited.")


# =============== CONSTANTS AND CONFIGURATION ===============

OPTICAL_CONSTANTS = {
    'hemoglobin': {
        'oxy_extinction_coefficients': {
            '660nm': 320.0, '700nm': 450.0, '730nm': 600.0, '760nm': 700.0, '805nm': 800.0,
            '850nm': 900.0, '900nm': 1000.0, '940nm': 1100.0, '1000nm': 1050.0
        },
        'deoxy_extinction_coefficients': {
            '660nm': 3200.0, '700nm': 1800.0, '730nm': 1200.0, '760nm': 900.0, '805nm': 800.0,
            '850nm': 750.0, '900nm': 700.0, '940nm': 650.0, '1000nm': 600.0
        },
        'molecular_weight_hb': 64500.0, # g/mol for tetrameric hemoglobin
        'concentration_unit_conversion': 10.0 # e.g., from g/dL to M (M = (g/dL) / MW * 10) -> (g/L) / MW
    },
    'glucose': {
        'refractive_index_change_per_mg_dl': 1.516e-5,
        'absorption_peaks_nm': [940, 1030, 1126, 1370, 1450, 1550, 1750, 2100, 2270, 2350],
        'specific_absorption_cm_inv_per_mg_dl': {
            '1550nm': 0.00023, '1750nm': 0.00015, '2100nm': 0.0008, '2270nm': 0.0005
        }
    },
    'total_cholesterol': {
        'scattering_coefficient_change_cm_inv_per_mg_dl': 0.0001, # This is highly illustrative
        'absorption_peaks_nm': [930, 1040, 1200, 1720, 1760, 2306, 2346], # C-H bonds
        'specific_absorption_cm_inv_per_mg_dl': {
             '1720nm': 0.0001, '2306nm': 0.00015 # Highly illustrative
        }
    },
    'ldl_cholesterol': { # Similar to total_cholesterol but potentially different magnitudes
        'scattering_coefficient_change_cm_inv_per_mg_dl': 0.00012,
        'specific_absorption_cm_inv_per_mg_dl': {'1720nm': 0.00012, '2306nm': 0.00018}
    },
    'hdl_cholesterol': {
        'scattering_coefficient_change_cm_inv_per_mg_dl': 0.00008,
        'specific_absorption_cm_inv_per_mg_dl': {'1720nm': 0.00008, '2306nm': 0.00012}
    },
    'triglycerides': {
        'scattering_coefficient_change_cm_inv_per_mg_dl': 0.00018,
        'absorption_peaks_nm': [930, 1150, 1210, 1390, 1725, 1765, 2310, 2350], # Ester and C-H bonds
        'specific_absorption_cm_inv_per_mg_dl': {
            '1725nm': 0.0002, '2310nm': 0.00025 # Highly illustrative
        }
    },
    'water': {
        'absorption_peaks_nm': [760, 970, 1190, 1450, 1940],
        'extinction_coefficients_cm_inv': { # Base-10 extinction for pure water
             '970nm': 0.48, '1450nm': 29.0, '1940nm': 120.0
        },
        'refractive_index_at_589nm': 1.333
    },
    'tissue_properties': {
        # Reduced scattering coefficient mu_s' = mu_s * (1-g)
        # mu_s'(lambda) = a * (lambda/lambda_ref)^-b (Approximation)
        'scattering_power_b': 1.2, # Typical range 0.5 - 1.5
        'scattering_amplitude_a_cm_inv': 15.0, # At reference wavelength (e.g., 500nm)
        'reference_wavelength_scattering_nm': 500.0,
        'anisotropy_g': 0.9,
        'baseline_refractive_index_medium': 1.36, # Dermis/interstitial fluid
        'blood_volume_fraction_typical': 0.02 # Range 0.01 - 0.05 in dermis
    }
}

DEFAULT_WAVELENGTHS_NM_STR = ['660nm', '805nm', '940nm', '1450nm', '1550nm', '1720nm', '2100nm']

# =============== UTILITY FUNCTIONS ===============
def _get_from_nested_dict(data_dict: Dict, map_list: List[str], default: Any = None) -> Any:
    """Access a nested dictionary item using a list of keys."""
    for k in map_list:
        if isinstance(data_dict, dict) and k in data_dict:
            data_dict = data_dict[k]
        else:
            return default
    return data_dict

def _robust_zscore(series: pd.Series) -> pd.Series:
    """Calculate Z-score using median and Median Absolute Deviation (MAD)."""
    median_val = series.median()
    mad_val = median_abs_deviation(series, nan_policy='omit')
    if mad_val == 0: # Avoid division by zero if all values are the same
        return pd.Series(np.zeros_like(series), index=series.index)
    # Scale MAD by 1.4826 to make it a consistent estimator for standard deviation for normal data
    return (series - median_val) / (1.4826 * mad_val)

def _filter_signal(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4,
                  filter_type: str = 'bandpass') -> np.ndarray:
    """Apply a Butterworth filter to the signal."""
    nyquist = 0.5 * fs
    if filter_type == 'bandpass':
        low = lowcut / nyquist
        high = highcut / nyquist
        if high >= 1.0: # Ensure high is less than 1.0 for nyquist
            high = 0.99
        if low <= 0.0: # Ensure low is greater than 0.0
            low = 1e-6
        if low >= high: # Ensure low < high
             logging.warning(f"Lowcut {lowcut}Hz is >= highcut {highcut}Hz. Returning original signal.")
             return data
        b, a = scipy_signal.butter(order, [low, high], btype='band', analog=False)
    elif filter_type == 'lowpass':
        high = highcut / nyquist
        if high >= 1.0: high = 0.99
        b, a = scipy_signal.butter(order, high, btype='low', analog=False)
    elif filter_type == 'highpass':
        low = lowcut / nyquist
        if low <= 0.0: low = 1e-6
        b, a = scipy_signal.butter(order, low, btype='high', analog=False)
    else:
        raise ValueError("filter_type must be 'bandpass', 'lowpass', or 'highpass'")
    
    # Use filtfilt for zero-phase filtering
    try:
        # Ensure data is long enough for the filter
        # The minimum length for filtfilt is max(len(a), len(b)) * 3, or a bit more practically.
        # For a 4th order Butterworth, len(a) and len(b) are order+1 = 5. So, > 15 samples.
        if len(data) > 3 * (order + 1) :
            filtered_data = scipy_signal.filtfilt(b, a, data)
            return filtered_data
        else:
            logging.warning(f"Signal length {len(data)} too short for filter order {order}. Returning original signal.")
            return data
    except ValueError as e:
        logging.error(f"Error during filtering: {e}. Returning original signal. Data length: {len(data)}")
        return data


# =============== PHYSICS-INFORMED NEURAL NETWORK LAYERS ===============

class BeerLambertLawLayer(tf.keras.layers.Layer):
    """
    Implements Beer-Lambert Law: A_lambda = sum_i (epsilon_lambda_i * c_i * l_lambda * DPF_lambda).
    Predicts concentrations (c_i) from estimated absorbances (A_lambda).
    Accounts for wavelength-dependent path length (l_lambda) and Differential Pathlength Factor (DPF).
    """
    def __init__(self,
                 chromophore_configs: Dict[str, Dict[str, float]],
                 wavelengths_nm_str: List[str],
                 learnable_path_length_factor: bool = True,
                 initial_base_path_length_mm: float = 6.0, # Baseline path length in tissue
                 learnable_dpf: bool = True,
                 initial_dpf_values: Optional[Union[float, List[float]]] = None, # Can be scalar or per wavelength
                 l2_reg_lstsq: float = 0.01, # Regularization for least squares solver
                 **kwargs):
        super(BeerLambertLawLayer, self).__init__(**kwargs)
        self.chromophore_configs = chromophore_configs
        self.wavelengths_nm_str = wavelengths_nm_str
        self.num_wavelengths = len(wavelengths_nm_str)
        self.learnable_path_length_factor = learnable_path_length_factor
        self.initial_base_path_length_mm = initial_base_path_length_mm # in mm
        self.learnable_dpf = learnable_dpf
        self.l2_reg_lstsq = l2_reg_lstsq

        self.chromophore_names = list(self.chromophore_configs.keys())
        self.num_chromophores = len(self.chromophore_names)

        # Construct epsilon matrix (extinction coefficients): rows=wavelengths, cols=chromophores
        # Units of epsilon should be consistent with concentration and path length (e.g., cm^-1 M^-1)
        epsilon_matrix_list = []
        for wl_str in self.wavelengths_nm_str:
            row = [self.chromophore_configs[chromo_name].get(wl_str, 0.0) for chromo_name in self.chromophore_names]
            epsilon_matrix_list.append(row)
        self.epsilon_matrix = tf.constant(epsilon_matrix_list, dtype=tf.float32) # Shape: (num_wavelengths, num_chromophores)

        # Initialize DPF values
        if initial_dpf_values is None:
            # Default DPF values (can be literature-based for specific wavelengths, e.g., ~4-6 for NIR in tissue)
            self.initial_dpf_values = [6.0] * self.num_wavelengths # Example default
        elif isinstance(initial_dpf_values, float):
            self.initial_dpf_values = [initial_dpf_values] * self.num_wavelengths
        elif isinstance(initial_dpf_values, list) and len(initial_dpf_values) == self.num_wavelengths:
            self.initial_dpf_values = initial_dpf_values
        else:
            raise ValueError("initial_dpf_values must be None, float, or list of len num_wavelengths")

    def build(self, input_shape_absorbances):
        # input_shape_absorbances: (batch_size, num_wavelengths)
        if input_shape_absorbances[-1] != self.num_wavelengths:
            raise ValueError(f"Input absorbance feature dim {input_shape_absorbances[-1]} != num_wavelengths {self.num_wavelengths}")

        # Path length factor (learnable scalar multiplier for base_path_length)
        # Base path length is in mm, convert to cm for calculations if epsilon is in cm^-1
        self.base_path_length_cm = tf.constant(self.initial_base_path_length_mm / 10.0, dtype=tf.float32)
        if self.learnable_path_length_factor:
            self.path_length_factor = self.add_weight(
                name='path_length_factor',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(1.0), # Starts at 1.0 * base_path_length
                constraint=NonNeg(), # Path length factor should be non-negative
                trainable=True
            )
        else:
            self.path_length_factor = tf.constant(1.0, dtype=tf.float32)

        # Differential Pathlength Factor (DPF) - can be per wavelength
        if self.learnable_dpf:
            self.dpf_values = self.add_weight(
                name='dpf_values',
                shape=(self.num_wavelengths,),
                initializer=tf.keras.initializers.Constant(self.initial_dpf_values),
                constraint=NonNeg(), # DPFs are positive
                trainable=True
            )
        else:
            self.dpf_values = tf.constant(self.initial_dpf_values, dtype=tf.float32)
        
        super(BeerLambertLawLayer, self).build(input_shape_absorbances)

    def call(self, inputs_absorbance):
        # inputs_absorbance shape: (batch_size, num_wavelengths)
        # Effective path length per wavelength: L_eff_lambda = base_path_length_cm * path_length_factor * DPF_lambda
        effective_path_length_cm = self.base_path_length_cm * self.path_length_factor * self.dpf_values
        # effective_path_length_cm shape: (num_wavelengths,)

        # Construct the system matrix M_lambda_j = epsilon_lambda_j * L_eff_lambda
        # Need to reshape effective_path_length_cm for broadcasting with epsilon_matrix
        # epsilon_matrix: (num_wavelengths, num_chromophores)
        # effective_path_length_cm_reshaped: (num_wavelengths, 1)
        effective_path_length_cm_reshaped = tf.expand_dims(effective_path_length_cm, axis=1)
        
        # M_matrix shape: (num_wavelengths, num_chromophores)
        M_matrix = self.epsilon_matrix * effective_path_length_cm_reshaped
        
        # We want to solve M @ C = Absorbances_input for C (concentrations)
        # C is (batch_size, num_chromophores)
        # M_matrix is (num_wavelengths, num_chromophores)
        # inputs_absorbance is (batch_size, num_wavelengths)

        # tf.linalg.lstsq solves Ax = b. Here, A=M_matrix, b=inputs_absorbance (transposed per batch item)
        # To use lstsq efficiently for batches, we tile M_matrix or use broadcasting if supported
        # Tile M_matrix to match batch size of inputs_absorbance
        batch_size = tf.shape(inputs_absorbance)[0]
        batch_M_matrix = tf.tile(tf.expand_dims(M_matrix, axis=0), [batch_size, 1, 1])
        
        # Reshape inputs_absorbance to be (batch_size, num_wavelengths, 1)
        batch_absorbances_reshaped = tf.expand_dims(inputs_absorbance, axis=-1)
        
        # concentrations shape: (batch_size, num_chromophores, 1)
        concentrations = tf.linalg.lstsq(batch_M_matrix,
                                         batch_absorbances_reshaped,
                                         l2_regularizer=self.l2_reg_lstsq,
                                         fast=True) # Use QR if M >= N, SVD otherwise
        
        # Squeeze the last dimension: (batch_size, num_chromophores)
        concentrations_squeezed = tf.squeeze(concentrations, axis=-1)
        
        return concentrations_squeezed

    def get_config(self):
        config = super().get_config()
        config.update({
            "chromophore_configs": self.chromophore_configs,
            "wavelengths_nm_str": self.wavelengths_nm_str,
            "learnable_path_length_factor": self.learnable_path_length_factor,
            "initial_base_path_length_mm": self.initial_base_path_length_mm,
            "learnable_dpf": self.learnable_dpf,
            "initial_dpf_values": self.initial_dpf_values, # Storing initial, not learned ones
            "l2_reg_lstsq": self.l2_reg_lstsq,
        })
        return config


class ScatteringModelLayer(tf.keras.layers.Layer):
    """
    Models changes in scattering due to biomarkers (e.g., glucose, lipids) affecting refractive index or particle concentration.
    Input features should be sensitive to scattering changes.
    """
    def __init__(self,
                 biomarker_ri_effects: Optional[Dict[str, float]] = None, # {'glucose_mg_dl': 1.5e-5 RIU/(mg/dL)}
                 biomarker_scatter_effects: Optional[Dict[str, float]] = None, # {'lipids_mg_dl': 0.001 d(mu_s')/(mg/dL)}
                 base_medium_ri: float = OPTICAL_CONSTANTS['tissue_properties']['baseline_refractive_index_medium'],
                 learnable_sensitivities: bool = True,
                 internal_mlp_units: List[int] = [32, 16],
                 internal_mlp_activation: str = 'relu',
                 **kwargs):
        super(ScatteringModelLayer, self).__init__(**kwargs)
        self.biomarker_ri_effects = biomarker_ri_effects if biomarker_ri_effects else {}
        self.biomarker_scatter_effects = biomarker_scatter_effects if biomarker_scatter_effects else {}
        self.all_scattering_biomarkers = sorted(list(set(list(self.biomarker_ri_effects.keys()) +
                                                       list(self.biomarker_scatter_effects.keys()))))
        self.num_scattering_biomarkers = len(self.all_scattering_biomarkers)
        
        self.base_medium_ri = base_medium_ri
        self.learnable_sensitivities = learnable_sensitivities
        self.internal_mlp_units = internal_mlp_units
        self.internal_mlp_activation = internal_mlp_activation

        if self.num_scattering_biomarkers == 0:
            warn("ScatteringModelLayer initialized with no biomarker effects. Will act as pass-through or simple MLP.")

        # MLP to map input scattering features to an intermediate representation
        mlp_layers = []
        for units in self.internal_mlp_units:
            mlp_layers.append(Dense(units, activation=self.internal_mlp_activation))
            mlp_layers.append(BatchNormalization()) # Optional, but often helpful
            mlp_layers.append(Dropout(0.1))       # Optional
        
        # Final dense layer in MLP to output a signal per biomarker
        # This signal will be proportional to the concentration.
        mlp_layers.append(Dense(self.num_scattering_biomarkers, activation='linear', name='scattering_biomarker_signals'))
        self.mlp = Sequential(mlp_layers, name='scattering_feature_mapper')

    def build(self, input_shape_scattering_features):
        # input_shape_scattering_features: (batch_size, num_input_scattering_features)
        
        if self.learnable_sensitivities and self.num_scattering_biomarkers > 0:
            # These sensitivities act as scaling factors for the output of the MLP
            # to map it to actual concentration units, considering the physical effect rates.
            # Initialize sensitivities based on physical rates if possible (e.g., to 1.0 if MLP output is already scaled by rate)
            initial_sensitivities = []
            for biomarker_name in self.all_scattering_biomarkers:
                ri_rate = self.biomarker_ri_effects.get(biomarker_name, 0.0)
                scatter_rate = self.biomarker_scatter_effects.get(biomarker_name, 0.0)
                # Use a combined or dominant rate for initialization, or simply 1.0
                # The idea is that (MLP_output / sensitivity) = concentration
                # Or MLP_output * sensitivity = concentration, depending on definition.
                # Let's define: MLP_output * sensitivity = concentration
                # So, sensitivity could be initialized to 1.0 or to the inverse of the physical rate,
                # if the MLP learns to output something proportional to delta_RI or delta_mu_s.
                # For simplicity, initialize to 1.0 and let network learn.
                initial_sensitivities.append(1.0)

            self.biomarker_sensitivities = self.add_weight(
                name='scattering_biomarker_sensitivities',
                shape=(self.num_scattering_biomarkers,),
                initializer=tf.keras.initializers.Constant(initial_sensitivities),
                trainable=True,
                constraint=NonNeg() # Sensitivities should be positive
            )
        elif self.num_scattering_biomarkers > 0:
            # Fixed sensitivities (use physical rates directly if MLP outputs delta_RI/delta_mu_s like signals)
            # This part requires careful thought on how the MLP output relates to concentration.
            # If MLP outputs are "normalized effect signals", then sensitivity is 1.0.
            # If MLP outputs are concentrations, then no sensitivities needed here.
            # Let's assume MLP outputs a signal that needs to be scaled by a sensitivity to get concentration.
            fixed_sens_values = []
            for biomarker_name in self.all_scattering_biomarkers:
                 # Placeholder: Use the inverse of the physical rates if MLP directly estimates physical change
                 # Example: if MLP estimates delta_RI, sensitivity = 1 / ri_change_rate
                 # For now, set to 1.0, assuming MLP's output scale is learned.
                fixed_sens_values.append(1.0)
            self.biomarker_sensitivities = tf.constant(fixed_sens_values, dtype=tf.float32)

        super(ScatteringModelLayer, self).build(input_shape_scattering_features)

    def call(self, inputs_scattering_features):
        # MLP maps input features to intermediate signals for each scattering-sensitive biomarker
        biomarker_signals = self.mlp(inputs_scattering_features) # (batch_size, num_scattering_biomarkers)

        if self.num_scattering_biomarkers == 0:
            return biomarker_signals # Or an empty tensor / raise error

        # Scale these signals by sensitivities to get estimated concentrations
        # concentrations = biomarker_signals * self.biomarker_sensitivities (if sensitivity scales signal to concentration)
        # OR concentrations = biomarker_signals / self.biomarker_sensitivities (if sensitivity is like physical rate)
        # Let's use multiplication: concentrations = biomarker_signals * sensitivity_factor
        # The 'sensitivity_factor' represents how much the learned 'biomarker_signal' needs to be scaled
        # to match the true concentration.
        concentrations = biomarker_signals * self.biomarker_sensitivities
        
        return concentrations # (batch_size, num_scattering_biomarkers)

    def get_config(self):
        config = super().get_config()
        config.update({
            "biomarker_ri_effects": self.biomarker_ri_effects,
            "biomarker_scatter_effects": self.biomarker_scatter_effects,
            "base_medium_ri": self.base_medium_ri,
            "learnable_sensitivities": self.learnable_sensitivities,
            "internal_mlp_units": self.internal_mlp_units,
            "internal_mlp_activation": self.internal_mlp_activation,
        })
        return config


class PhysiologicalConstraintLayer(tf.keras.layers.Layer):
    """
    Enforces physiologically plausible ranges for biomarker predictions.
    The input to this layer should be the raw model prediction before scaling.
    """
    def __init__(self, biomarker_name: str, activation_type: str = 'scaled_sigmoid', **kwargs):
        super(PhysiologicalConstraintLayer, self).__init__(**kwargs)
        self.biomarker_name = biomarker_name.lower().replace(" ", "_")
        self.activation_type = activation_type.lower()
        
        # Define physiological ranges (min_val, max_val, typical_low, typical_high)
        # These ranges help in scaling activations appropriately.
        # Units must be consistent with model targets.
        self.physiological_params = {
            'glucose':              {'min': 30.0,  'max': 800.0, 'low': 70.0,  'high': 180.0, 'unit': 'mg/dL'},
            'spo2':                 {'min': 0.40,  'max': 1.001, 'low': 0.94,  'high': 0.99,  'unit': '%/100'}, # Max slightly > 1 for sigmoid
            'total_cholesterol':    {'min': 50.0,  'max': 600.0, 'low': 120.0, 'high': 200.0, 'unit': 'mg/dL'},
            'ldl_cholesterol':      {'min': 20.0,  'max': 400.0, 'low': 50.0,  'high': 130.0, 'unit': 'mg/dL'},
            'hdl_cholesterol':      {'min': 10.0,  'max': 150.0, 'low': 40.0,  'high': 60.0,  'unit': 'mg/dL'},
            'triglycerides':        {'min': 20.0,  'max': 1000.0,'low': 50.0,  'high': 150.0, 'unit': 'mg/dL'},
            'hemoglobin':           {'min': 4.0,   'max': 28.0,  'low': 12.0,  'high': 18.0,  'unit': 'g/dL'},
            # Add other biomarkers as needed
        }
        
        self.params = self.physiological_params.get(self.biomarker_name)
        if self.params is None:
            warn_msg = (f"Physiological parameters for biomarker '{self.biomarker_name}' not defined. "
                        f"Layer will apply a generic activation like '{self.activation_type}' if possible, or pass through.")
            warn(warn_msg)
            # For undefined biomarkers, we might allow a pass-through or a simple linear/relu activation
            # if the activation_type is generic. Otherwise, it's an issue.
            if self.activation_type not in ['linear', 'relu', 'elu', 'leakyrelu', 'selu', 'softplus', 'tanh', 'sigmoid']:
                 raise ValueError(f"Cannot apply constraint for unknown biomarker '{self.biomarker_name}' with activation '{self.activation_type}'")

    def call(self, inputs):
        # inputs are raw model outputs (logits) from the preceding layer
        if self.params is None: # Biomarker not in our defined list
            if self.activation_type == 'linear': return tf.keras.activations.linear(inputs)
            if self.activation_type == 'relu': return tf.keras.activations.relu(inputs)
            # Add more generic activations or just pass through
            logging.debug(f"PhysiologicalConstraintLayer: No params for {self.biomarker_name}, passing through input.")
            return inputs

        min_val, max_val = self.params['min'], self.params['max']
        # typical_low, typical_high = self.params['low'], self.params['high'] # Could be used for smarter scaling

        if self.activation_type == 'scaled_sigmoid':
            # Sigmoid maps to (0,1), then scale and shift
            output = tf.sigmoid(inputs) * (max_val - min_val) + min_val
        elif self.activation_type == 'scaled_tanh':
            # Tanh maps to (-1,1). (tanh(x)*0.5 + 0.5) maps to (0,1), then scale and shift
            output = (tf.tanh(inputs) * 0.5 + 0.5) * (max_val - min_val) + min_val
        elif self.activation_type == 'softplus_offset':
            # Softplus maps to (0, inf). Shifted softplus can map to (min_val, inf)
            # This provides a soft lower bound but no hard upper bound.
            # output = tf.keras.activations.softplus(inputs) + min_val
            # To also have a soft upper bound, one might use:
            # output = min_val + (max_val - min_val) * tf.sigmoid( (inputs - center_logit) / scale_logit)
            # For now, simple softplus for lower bound:
            output = tf.keras.activations.softplus(inputs) + min_val
            # Clip at max_val if using this for an upper bound too (less ideal than sigmoid/tanh for hard bounds)
            output = tf.clip_by_value(output, min_val, max_val)
        elif self.activation_type == 'clipped_linear':
            # Apply linear activation then clip. This is often less smooth.
            output = tf.keras.activations.linear(inputs)
            output = tf.clip_by_value(output, min_val, max_val)
        elif self.activation_type == 'custom_beta':
            # Use a Beta distribution transformed to the range [min_val, max_val]
            # The 'inputs' would need to be interpreted as parameters for the Beta dist (alpha, beta)
            # This is more complex and suitable if the output layer is probabilistic.
            # For a deterministic layer, this is an over-complication.
            # Assuming inputs are logits for a scaled sigmoid for simplicity here.
            warn("Custom_beta for PhysiologicalConstraintLayer is advanced and assumes probabilistic output. Falling back to scaled_sigmoid.")
            output = tf.sigmoid(inputs) * (max_val - min_val) + min_val
        else:
            # Fallback for generic Keras activations if specified, without range scaling
            try:
                activation_fn = tf.keras.activations.get(self.activation_type)
                output = activation_fn(inputs)
                # We could attempt to clip this too, but it might not be what the user intends
                # if they chose a generic activation.
                logging.warning(f"Generic activation '{self.activation_type}' used for {self.biomarker_name}. "
                                f"Output will not be strictly constrained to [{min_val}, {max_val}] by this layer's scaling.")
            except ValueError:
                raise ValueError(f"Unsupported activation_type: {self.activation_type} for PhysiologicalConstraintLayer")
        
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "biomarker_name": self.biomarker_name,
            "activation_type": self.activation_type,
        })
        return config

# ... (Continuing from the previous code block, including imports, constants, physics layers, and BaseSignalTransformer) ...

# =============== SIGNAL PREPROCESSING AND SEGMENTATION ===============

class PPGSignalPreprocessor(BaseSignalTransformer):
    """
    Preprocesses raw PPG signals from input DataFrame columns.
    Operations include: type conversion, outlier capping, filtering,
    detrending, normalization, and segmentation.
    Outputs a DataFrame where each row can represent a segment if segmentation is enabled,
    or a processed full signal if not.
    """
    def __init__(self,
                 signal_column_prefixes: List[str] = ['red', 'ir', 'green', 'blue'],
                 expected_input_type: type = np.ndarray, # or list
                 sampling_rate_hz: float = 100.0,
                 outlier_capping_method: Optional[str] = 'iqr', # 'iqr', 'zscore', 'robust_zscore', None
                 outlier_threshold: float = 3.0, # For zscore methods: std devs, for iqr: multiplier
                 filter_type: Optional[str] = 'bandpass',
                 lowcut_hz: float = 0.4,
                 highcut_hz: float = 10.0, # Wider for some features like dicrotic notch
                 filter_order: int = 4,
                 detrend_method: Optional[str] = 'polynomial',
                 poly_detrend_degree: int = 3, # Higher degree for more complex baseline wander
                 sg_detrend_window_s: float = 1.5, # Savitzky-Golay window for detrending
                 sg_detrend_polyorder: int = 2,
                 normalization_method: Optional[str] = 'zscore', # 'zscore', 'robust_zscore', 'minmax', 'maxabs', None
                 segment_length_s: Optional[float] = 8.0, # Longer segments can be better for HRV, non-linear
                 segment_overlap_ratio: float = 0.75, # Higher overlap for more data augmentation
                 min_segment_quality_threshold: Optional[float] = None, # Placeholder for a quality check
                 output_prefix: str = 'proc_',
                 keep_original_columns: bool = False): # Whether to keep non-signal columns
        super().__init__()
        self.signal_column_prefixes = signal_column_prefixes
        self.expected_input_type = expected_input_type
        self.sampling_rate_hz = sampling_rate_hz
        self.outlier_capping_method = outlier_capping_method
        self.outlier_threshold = outlier_threshold
        self.filter_type = filter_type
        self.lowcut_hz = lowcut_hz
        self.highcut_hz = highcut_hz
        self.filter_order = filter_order
        self.detrend_method = detrend_method
        self.poly_detrend_degree = poly_detrend_degree
        self.sg_detrend_window_s = sg_detrend_window_s
        self.sg_detrend_polyorder = sg_detrend_polyorder
        self.normalization_method = normalization_method
        self.segment_length_s = segment_length_s
        self.segment_overlap_ratio = segment_overlap_ratio
        self.min_segment_quality_threshold = min_segment_quality_threshold
        self.output_prefix = output_prefix
        self.keep_original_columns = keep_original_columns

        if self.segment_length_s is not None:
            self.segment_length_samples = int(self.segment_length_s * self.sampling_rate_hz)
            if self.segment_length_samples <= 0:
                raise ValueError("segment_length_s and sampling_rate_hz must result in > 0 samples.")
            self.segment_step_samples = int(self.segment_length_samples * (1.0 - self.segment_overlap_ratio))
            if self.segment_step_samples <= 0:
                logging.info("Segment step samples is <= 0 due to high overlap or short segment. Setting to 1.")
                self.segment_step_samples = 1
        else:
            self.segment_length_samples = None
            self.segment_step_samples = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        super().fit(X, y)
        self.input_signal_columns_ = [col for col in X.columns if any(col.startswith(p) for p in self.signal_column_prefixes)]
        if not self.input_signal_columns_:
            warn(f"PPGSignalPreprocessor: No signal columns found with prefixes {self.signal_column_prefixes}. Ensure DataFrame has these columns.")
        
        # Define output feature names (processed signal columns)
        self.created_feature_names_ = [f"{self.output_prefix}{col_name}" for col_name in self.input_signal_columns_]
        
        # If segmenting, additional metadata columns will be added to the output DataFrame
        self.metadata_columns_ = []
        if self.segment_length_s is not None:
             self.metadata_columns_ = ['original_index', 'segment_id', 'segment_start_time_s']
        
        # Store non-signal columns if they are to be kept
        self.non_signal_columns_ = [col for col in X.columns if col not in self.input_signal_columns_]

        return self

    def _cap_outliers(self, signal_array: np.ndarray) -> np.ndarray:
        if self.outlier_capping_method is None or len(signal_array) < 10: # Need enough points for stats
            return signal_array
        
        s = pd.Series(signal_array)
        if self.outlier_capping_method == 'iqr':
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr_val = q3 - q1
            lower_bound = q1 - self.outlier_threshold * iqr_val
            upper_bound = q3 + self.outlier_threshold * iqr_val
        elif self.outlier_capping_method == 'zscore':
            mean_val = s.mean()
            std_val = s.std()
            if std_val == 0: return signal_array # All values are same
            lower_bound = mean_val - self.outlier_threshold * std_val
            upper_bound = mean_val + self.outlier_threshold * std_val
        elif self.outlier_capping_method == 'robust_zscore':
            median_val = s.median()
            mad_val = median_abs_deviation(s, nan_policy='omit', scale='normal') # scale='normal' makes it estimate std
            if mad_val == 0: return signal_array # All values are same or MAD is zero
            lower_bound = median_val - self.outlier_threshold * mad_val
            upper_bound = median_val + self.outlier_threshold * mad_val
        else:
            return signal_array
        
        return np.clip(signal_array, lower_bound, upper_bound)

    def _process_single_signal(self, signal_data: Union[np.ndarray, list]) -> Optional[np.ndarray]:
        # 0. Type Conversion and Validation
        if not isinstance(signal_data, self.expected_input_type):
            try:
                if self.expected_input_type == np.ndarray:
                    signal_array = np.array(signal_data, dtype=float)
                else: # list
                    signal_array = np.array(list(signal_data), dtype=float) # Ensure it's a list of numbers
            except Exception as e:
                logging.warning(f"Could not convert signal data to np.ndarray: {e}. Skipping this signal.")
                return None
        else:
            signal_array = np.array(signal_data, dtype=float) # Ensure float for processing

        if signal_array.ndim != 1:
            logging.warning(f"Signal data is not 1D (shape: {signal_array.shape}). Skipping.")
            return None
        if len(signal_array) < 10: # Arbitrary short signal threshold
            logging.warning(f"Signal too short (length: {len(signal_array)}). Skipping.")
            return None
        
        processed_signal = np.copy(signal_array)

        # 1. Outlier Capping
        processed_signal = self._cap_outliers(processed_signal)

        # 2. Filtering
        if self.filter_type and len(processed_signal) > 3 * (self.filter_order + 1):
            processed_signal = _filter_signal(processed_signal, self.lowcut_hz, self.highcut_hz,
                                              self.sampling_rate_hz, self.filter_order, self.filter_type)
        
        # 3. Detrending
        if self.detrend_method:
            if self.detrend_method == 'polynomial' and len(processed_signal) > self.poly_detrend_degree:
                x_axis = np.arange(len(processed_signal))
                try:
                    coeffs = np.polyfit(x_axis, processed_signal, self.poly_detrend_degree)
                    trend = np.polyval(coeffs, x_axis)
                    processed_signal = processed_signal - trend
                except (np.linalg.LinAlgError, ValueError) as e: # Handle potential polyfit errors
                    logging.warning(f"Polynomial detrending failed: {e}. Skipping detrending.")

            elif self.detrend_method == 'sg_filter':
                window_samples = int(self.sg_detrend_window_s * self.sampling_rate_hz)
                if window_samples % 2 == 0: window_samples += 1 # Must be odd
                if len(processed_signal) > window_samples and window_samples > self.sg_detrend_polyorder:
                    try:
                        trend = scipy_signal.savgol_filter(processed_signal, window_samples, self.sg_detrend_polyorder)
                        processed_signal = processed_signal - trend
                    except ValueError as e:
                        logging.warning(f"Savitzky-Golay detrending failed: {e}. Skipping detrending.")
                # else: not enough data for SG trend removal or invalid params

        # 4. Normalization (applied per signal/segment)
        if self.normalization_method:
            if np.all(np.isfinite(processed_signal)): # Ensure no NaNs/Infs before normalization
                if self.normalization_method == 'zscore':
                    mean_val, std_val = np.mean(processed_signal), np.std(processed_signal)
                    if std_val > 1e-8: # Avoid division by zero
                        processed_signal = (processed_signal - mean_val) / std_val
                    else: # All values are (nearly) the same
                        processed_signal = np.zeros_like(processed_signal)
                elif self.normalization_method == 'robust_zscore':
                    median_val = np.median(processed_signal)
                    mad_val = median_abs_deviation(processed_signal, scale='normal', nan_policy='omit')
                    if mad_val > 1e-8:
                        processed_signal = (processed_signal - median_val) / mad_val
                    else:
                        processed_signal = np.zeros_like(processed_signal)
                elif self.normalization_method == 'minmax':
                    min_val, max_val = np.min(processed_signal), np.max(processed_signal)
                    if (max_val - min_val) > 1e-8:
                        processed_signal = (processed_signal - min_val) / (max_val - min_val)
                    else:
                        processed_signal = np.full_like(processed_signal, 0.5) # Mid-point if range is zero
                elif self.normalization_method == 'maxabs':
                    max_abs_val = np.max(np.abs(processed_signal))
                    if max_abs_val > 1e-8:
                        processed_signal = processed_signal / max_abs_val
                    else:
                        processed_signal = np.zeros_like(processed_signal)
            else:
                logging.warning("Signal contains non-finite values before normalization. Skipping normalization.")
        
        return processed_signal

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        super().transform(X) # Basic checks from base class

        if not self.input_signal_columns_:
            # If no signal columns were identified during fit, return X as is (or only non-signal columns if specified)
            return X[self.non_signal_columns_] if self.keep_original_columns and self.non_signal_columns_ else X

        all_processed_segments_data = []

        for original_idx, row in X.iterrows():
            processed_signals_for_row = {}
            valid_row = True
            for signal_col_name in self.input_signal_columns_:
                raw_signal_data = row.get(signal_col_name)
                if raw_signal_data is None:
                    logging.warning(f"Signal column '{signal_col_name}' not found in row {original_idx}. Skipping.")
                    valid_row = False; break
                
                processed_full_signal = self._process_single_signal(raw_signal_data)
                if processed_full_signal is None:
                    valid_row = False; break
                processed_signals_for_row[f"{self.output_prefix}{signal_col_name}"] = processed_full_signal
            
            if not valid_row:
                continue

            # Handle segmentation
            if self.segment_length_samples is not None:
                # Check if all processed signals for this row are long enough for at least one segment
                min_len = min(len(s) for s in processed_signals_for_row.values())
                if min_len < self.segment_length_samples:
                    logging.debug(f"Row {original_idx}: shortest processed signal ({min_len}) too short for segmentation ({self.segment_length_samples}). Skipping.")
                    continue

                # Assume all signals in processed_signals_for_row have same length after initial check,
                # or handle varying lengths if necessary (more complex). For simplicity, assume consistent length for segmentation.
                # Reference length for segmentation:
                ref_signal_len = len(next(iter(processed_signals_for_row.values())))

                for i in range(0, ref_signal_len - self.segment_length_samples + 1, self.segment_step_samples):
                    segment_data = {}
                    is_valid_segment = True
                    for proc_col_name, full_signal in processed_signals_for_row.items():
                        if i + self.segment_length_samples <= len(full_signal): # Ensure segment is within bounds for this channel too
                            segment = full_signal[i : i + self.segment_length_samples]
                            # Optionally, re-normalize segment if normalization was per-signal and not per-segment earlier
                            # For now, assume normalization was sufficient or will be handled by downstream tasks
                            segment_data[proc_col_name] = segment
                        else:
                            is_valid_segment = False; break # Mismatched lengths among processed signals
                    
                    if is_valid_segment:
                        # Add metadata
                        segment_data['original_index'] = original_idx
                        segment_data['segment_id'] = i // self.segment_step_samples
                        segment_data['segment_start_time_s'] = i / self.sampling_rate_hz
                        
                        # Add non-signal columns if keep_original_columns
                        if self.keep_original_columns:
                            for ns_col in self.non_signal_columns_:
                                segment_data[ns_col] = row.get(ns_col)
                        all_processed_segments_data.append(segment_data)
            else: # No segmentation, use the full processed signals
                full_signal_data = {}
                full_signal_data['original_index'] = original_idx # Keep track of original row
                for proc_col_name, signal_array in processed_signals_for_row.items():
                    full_signal_data[proc_col_name] = signal_array # Store the array itself
                
                if self.keep_original_columns:
                    for ns_col in self.non_signal_columns_:
                        full_signal_data[ns_col] = row.get(ns_col)
                all_processed_segments_data.append(full_signal_data)

        if not all_processed_segments_data:
            logging.warning("PPGSignalPreprocessor: No data produced after processing and segmentation. Returning empty DataFrame.")
            # Return DataFrame with expected columns, even if empty
            expected_cols = self.metadata_columns_ + self.created_feature_names_
            if self.keep_original_columns:
                expected_cols += self.non_signal_columns_
            return pd.DataFrame(columns=list(dict.fromkeys(expected_cols))) # Ensure unique columns

        output_df = pd.DataFrame(all_processed_segments_data)
        # Reorder columns to have metadata first, then processed signals, then original other columns
        ordered_cols = self.metadata_columns_ + self.created_feature_names_
        if self.keep_original_columns:
            ordered_cols += self.non_signal_columns_
        
        # Ensure all expected columns are present, fill with NaN if some are missing (should not happen with current logic)
        final_cols = []
        for col in ordered_cols:
            if col in output_df.columns:
                final_cols.append(col)
            # else: # This case might indicate an issue if a column is expected but not generated
            #    output_df[col] = np.nan 
            #    final_cols.append(col)
        
        return output_df[list(dict.fromkeys(final_cols))] # Use unique column names in specified order

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if not self._is_fitted:
             warn("get_feature_names_out called before fit. Defaulting based on init params.")
             # Attempt to construct from init params if not fitted (e.g., for ColumnTransformer inspection)
             temp_input_signal_cols = [f"{p}_dummy" for p in self.signal_column_prefixes] # Dummy input signal names
             created_names = [f"{self.output_prefix}{col}" for col in temp_input_signal_cols]
             meta_names = []
             if self.segment_length_s is not None:
                 meta_names = ['original_index', 'segment_id', 'segment_start_time_s']
             
             original_other_cols = []
             if input_features is not None: # input_features are all columns given to fit/transform
                 original_other_cols = [col for col in input_features if not any(col.startswith(p) for p in self.signal_column_prefixes)]
             
             if self.keep_original_columns:
                 return list(dict.fromkeys(meta_names + created_names + original_other_cols))
             else:
                 return list(dict.fromkeys(meta_names + created_names))

        # If fitted:
        output_cols = self.metadata_columns_ + self.created_feature_names_
        if self.keep_original_columns:
            output_cols += self.non_signal_columns_
        
        return list(dict.fromkeys(output_cols)) # Return unique column names in order


# =============== ADVANCED FEATURE EXTRACTORS ===============

class WaveletFeatureExtractor(BaseSignalTransformer):
    """
    Extracts wavelet-based time-frequency features from processed PPG signals.
    Assumes input DataFrame X has columns containing processed signal arrays (e.g., from PPGSignalPreprocessor).
    """
    def __init__(self,
                 signal_column_glob: str = "proc_*", # Glob pattern to identify processed signal columns
                 wavelet_name: str = 'db8', # Daubechies 8, good for biomedical signals
                 max_level: Optional[int] = None, # If None, pywt.dwt_max_level is used
                 decomp_mode: str = 'symmetric', # PyWavelets mode for signal extension
                 features_to_extract: List[str] = [
                     'energy', 'entropy', 'std', 'mean', 'median', 'iqr', 'kurtosis', 'skew',
                     'zero_crossings', 'mean_crossings', 'shannon_entropy',
                     'approx_coeff_stats', # Stats from approximation coefficients at max_level
                     'detail_coeff_ratios' # Ratios of energy/std between adjacent detail levels
                 ],
                 output_prefix_per_signal_col: bool = True): # If True, prepends original signal col name to features
        super().__init__()
        if not has_pywavelets:
            raise ImportError("PyWavelets library is required for WaveletFeatureExtractor but not installed.")
        self.signal_column_glob = signal_column_glob # Used to find signal columns if not explicitly set
        self.wavelet_name = wavelet_name
        self.max_level = max_level
        self.decomp_mode = decomp_mode
        self.features_to_extract = features_to_extract
        self.output_prefix_per_signal_col = output_prefix_per_signal_col
        
        try:
            self.wavelet = pywt.Wavelet(self.wavelet_name)
        except ValueError as e:
            raise ValueError(f"Invalid wavelet_name '{self.wavelet_name}': {e}")

    def _get_signal_columns(self, X: pd.DataFrame) -> List[str]:
        # More robust way to find signal columns using glob-like matching if needed
        # For now, simple startswith based on the output of PPGSignalPreprocessor
        if self.signal_column_glob == "proc_*": # Common case
            return [col for col in X.columns if col.startswith("proc_") and isinstance(X[col].iloc[0], (np.ndarray, list))]
        else:
            # Implement more complex glob matching if necessary, or require explicit list
            # This is a placeholder for more advanced column selection
            return [col for col in X.columns if col.startswith(self.signal_column_glob.replace("*","")) and isinstance(X[col].iloc[0], (np.ndarray, list))]


    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        super().fit(X,y)
        self.input_signal_columns_ = self._get_signal_columns(X)
        if not self.input_signal_columns_:
            warn(f"WaveletFeatureExtractor: No signal columns found matching glob '{self.signal_column_glob}'.")

        # Dynamically generate created_feature_names_ based on features_to_extract and signal columns
        # This is crucial for get_feature_names_out
        temp_created_names = []
        for signal_col_name in self.input_signal_columns_: # Use identified signal columns
            prefix = f"{signal_col_name}_wl_" if self.output_prefix_per_signal_col else "wl_"
            
            # Simulate decomposition up to a reasonable level for naming
            # This doesn't run actual decomposition, just estimates levels for naming
            # Actual levels determined per signal in transform
            num_levels_for_naming = self.max_level if self.max_level is not None else 5 # Example fixed for naming
            
            if 'approx_coeff_stats' in self.features_to_extract:
                for stat in ['energy', 'entropy', 'std', 'mean', 'median', 'iqr', 'kurtosis', 'skew']:
                    temp_created_names.append(f"{prefix}app_L{num_levels_for_naming}_{stat}")

            for level in range(1, num_levels_for_naming + 1):
                level_prefix = f"{prefix}det_L{level}_"
                if 'energy' in self.features_to_extract: temp_created_names.append(f"{level_prefix}energy")
                if 'entropy' in self.features_to_extract: temp_created_names.append(f"{level_prefix}entropy")
                # ... add all other stats from features_to_extract
                for stat_feature in ['std', 'mean', 'median', 'iqr', 'kurtosis', 'skew', 'zero_crossings', 'mean_crossings', 'shannon_entropy']:
                     if stat_feature in self.features_to_extract:
                         temp_created_names.append(f"{level_prefix}{stat_feature}")

            if 'detail_coeff_ratios' in self.features_to_extract:
                for level in range(1, num_levels_for_naming): # Ratios up to L(N-1)/LN
                    temp_created_names.append(f"{prefix}det_energy_ratio_L{level}_L{level+1}")
                    temp_created_names.append(f"{prefix}det_std_ratio_L{level}_L{level+1}")
        
        self.created_feature_names_ = sorted(list(set(temp_created_names))) # Ensure unique and sorted
        self.non_signal_columns_ = [col for col in X.columns if col not in self.input_signal_columns_]
        return self

    def _calculate_entropy(self, coeffs: np.ndarray, method: str = 'shannon_probability') -> float:
        if len(coeffs) == 0: return 0.0
        if method == 'shannon_probability':
            # Probability distribution from normalized squared coefficients (energy)
            prob_dist = (coeffs ** 2) / (np.sum(coeffs ** 2) + 1e-9)
            prob_dist = prob_dist[prob_dist > 1e-9] # Avoid log(0)
            return -np.sum(prob_dist * np.log2(prob_dist))
        elif method == 'log_energy': # Another common definition
            # Sum of log of squared coeffs (needs careful handling of zeros/negatives if raw coeffs)
            # Ensure coeffs are positive for log, e.g., use energy
            energy_coeffs = coeffs**2
            energy_coeffs = energy_coeffs[energy_coeffs > 1e-9]
            if len(energy_coeffs) == 0: return 0.0
            return np.sum(np.log2(energy_coeffs))
        else:
            return 0.0
            
    def _extract_coeff_features(self, coeffs: np.ndarray, prefix: str) -> Dict[str, float]:
        features = {}
        if len(coeffs) == 0: # Handle empty coefficients (e.g., from very short signals)
            for stat_feature in self.features_to_extract: # Create keys with 0 value
                if stat_feature in ['energy', 'entropy', 'std', 'mean', 'median', 'iqr', 'kurtosis', 'skew',
                                    'zero_crossings', 'mean_crossings', 'shannon_entropy']:
                     features[f"{prefix}{stat_feature}"] = 0.0
            return features

        if 'energy' in self.features_to_extract: features[f"{prefix}energy"] = np.sum(coeffs**2)
        if 'entropy' in self.features_to_extract: features[f"{prefix}entropy"] = self._calculate_entropy(coeffs, 'shannon_probability')
        if 'shannon_entropy' in self.features_to_extract: features[f"{prefix}shannon_entropy"] = self._calculate_entropy(coeffs, 'shannon_probability')
        
        if 'std' in self.features_to_extract: features[f"{prefix}std"] = np.std(coeffs)
        if 'mean' in self.features_to_extract: features[f"{prefix}mean"] = np.mean(coeffs)
        if 'median' in self.features_to_extract: features[f"{prefix}median"] = np.median(coeffs)
        if 'iqr' in self.features_to_extract: features[f"{prefix}iqr"] = iqr(coeffs)
        if 'kurtosis' in self.features_to_extract: features[f"{prefix}kurtosis"] = kurtosis(coeffs, fisher=True, bias=False) # Fisher (normal=0), unbiased
        if 'skew' in self.features_to_extract: features[f"{prefix}skew"] = skew(coeffs, bias=False) # Unbiased

        if 'zero_crossings' in self.features_to_extract:
            features[f"{prefix}zero_crossings"] = len(np.where(np.diff(np.sign(coeffs)))[0])
        if 'mean_crossings' in self.features_to_extract:
            mean_val = np.mean(coeffs)
            features[f"{prefix}mean_crossings"] = len(np.where(np.diff(np.sign(coeffs - mean_val)))[0])
        return features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        super().transform(X) # Basic checks
        
        if not self.input_signal_columns_:
            return X[self.non_signal_columns_] if self.non_signal_columns_ else X # Return non-signal or original

        all_row_features = []

        for _, row in X.iterrows():
            row_features = {}
            # Keep non-signal columns from the input row
            for col_name in self.non_signal_columns_:
                row_features[col_name] = row[col_name]

            for signal_col_name in self.input_signal_columns_:
                signal_array = row.get(signal_col_name)
                if signal_array is None or not isinstance(signal_array, (np.ndarray, list)) or len(signal_array) < self.wavelet.dec_len:
                    logging.debug(f"Signal '{signal_col_name}' is invalid or too short for wavelet decomposition. Skipping.")
                    # Add NaNs or zeros for expected features from this signal to maintain DataFrame structure
                    # This is complex; for now, skip adding features for this problematic signal in this row.
                    # Better: pre-fill row_features with NaNs for all expected wl features then update.
                    continue
                
                signal_array = np.asarray(signal_array, dtype=float)

                current_max_level = self.max_level
                if current_max_level is None:
                    current_max_level = pywt.dwt_max_level(len(signal_array), self.wavelet)
                if current_max_level <= 0: # Not enough data for any level
                    logging.debug(f"Signal '{signal_col_name}' length {len(signal_array)} too short for any wavelet decomposition level with {self.wavelet.name}. Max level {current_max_level}.")
                    continue

                try:
                    coeffs_list = pywt.wavedec(signal_array, self.wavelet, level=current_max_level, mode=self.decomp_mode)
                except Exception as e:
                    logging.error(f"Wavelet decomposition failed for {signal_col_name}: {e}")
                    continue

                # Features from approximation coefficients (cA)
                cA = coeffs_list[0]
                prefix_base = f"{signal_col_name}_wl_" if self.output_prefix_per_signal_col else "wl_"
                
                if 'approx_coeff_stats' in self.features_to_extract:
                    approx_prefix = f"{prefix_base}app_L{current_max_level}_"
                    row_features.update(self._extract_coeff_features(cA, approx_prefix))

                # Features from detail coefficients (cD)
                detail_coeffs_all_levels = coeffs_list[1:]
                level_energies = []
                level_stds = []

                for level_idx, cD in enumerate(reversed(detail_coeffs_all_levels)): # cD_L1 is last in list
                    level_num = level_idx + 1 # Actual level number (1 to max_level)
                    detail_prefix = f"{prefix_base}det_L{level_num}_"
                    level_features = self._extract_coeff_features(cD, detail_prefix)
                    row_features.update(level_features)
                    
                    if 'detail_coeff_ratios' in self.features_to_extract:
                        level_energies.append(level_features.get(f"{detail_prefix}energy", 0.0))
                        level_stds.append(level_features.get(f"{detail_prefix}std", 0.0))
                
                # Ratios between adjacent detail levels
                if 'detail_coeff_ratios' in self.features_to_extract:
                    for i in range(len(level_energies) - 1):
                        # level_energies[i] is for L(i+1), level_energies[i+1] is for L(i+2)
                        ratio_energy_key = f"{prefix_base}det_energy_ratio_L{i+1}_L{i+2}"
                        row_features[ratio_energy_key] = self._safe_division(level_energies[i], level_energies[i+1])
                        
                        ratio_std_key = f"{prefix_base}det_std_ratio_L{i+1}_L{i+2}"
                        row_features[ratio_std_key] = self._safe_division(level_stds[i], level_stds[i+1])
            
            all_row_features.append(row_features)

        if not all_row_features:
            logging.warning("WaveletFeatureExtractor: No features extracted. Input might be empty or all signals invalid.")
            # Return original non-signal columns or an empty DataFrame with expected feature names
            if self.non_signal_columns_:
                 # This is tricky if X was empty to begin with
                if X.empty:
                    return pd.DataFrame(columns=self.non_signal_columns_ + self.created_feature_names_)
                else:
                    return X[self.non_signal_columns_].copy() # Return only non-signal columns
            else: # No non-signal columns to keep, and no features extracted
                return pd.DataFrame(columns=self.created_feature_names_)


        features_df = pd.DataFrame(all_row_features)
        # Fill any features that were expected (from fit) but not generated (e.g., due to short signals) with 0 or NaN
        for expected_feature_name in self.created_feature_names_:
            if expected_feature_name not in features_df.columns:
                features_df[expected_feature_name] = 0.0 # Or np.nan

        # Ensure correct column order and include non_signal_columns if they were part of all_row_features
        final_columns = self.non_signal_columns_ + self.created_feature_names_
        final_columns = [col for col in final_columns if col in features_df.columns] # Only existing columns
        
        return features_df[list(dict.fromkeys(final_columns))].fillna(0.0) # Fill NaNs with 0, ensure unique cols

# ... (Continuing from the previous code block, including SpectralFeatureExtractor) ...

class HeartRateVariabilityTransformer(BaseSignalTransformer):
    """
    Extracts Heart Rate Variability (HRV) features from processed PPG signals.
    This requires reliable peak detection to obtain RR intervals (or PP intervals for PPG).
    Features include time-domain, frequency-domain (from Lomb-Scargle periodogram of RRIs),
    and some non-linear HRV metrics.
    """
    def __init__(self,
                 signal_column_glob: str = "proc_*", # Processed signal to find peaks from (e.g., IR channel)
                 sampling_rate_hz: float = 100.0,
                 peak_detection_method: str = 'simple_scipy', # 'simple_scipy', 'heartpy' (if available)
                 # Parameters for simple_scipy peak detection
                 min_peak_height_factor: float = 0.5, # Min height = factor * (max - min of signal)
                 min_peak_distance_s: float = 0.3, # Min distance between peaks in seconds
                 # HRV Features
                 time_domain_features: List[str] = ['mean_rr', 'sdnn', 'rmssd', 'pnn50', 'hr_mean', 'hr_std'],
                 freq_domain_features: List[str] = ['lf_power', 'hf_power', 'lf_hf_ratio', 'total_power_hrv'],
                 freq_bands_hrv_hz: Dict[str, Tuple[float, float]] = { # Standard HRV bands
                     'ulf': (0.0, 0.003),    # Ultra Low Frequency (requires long signals)
                     'vlf': (0.003, 0.04),  # Very Low Frequency
                     'lf': (0.04, 0.15),    # Low Frequency (sympathetic/parasympathetic)
                     'hf': (0.15, 0.4),     # High Frequency (parasympathetic - respiration)
                 },
                 nonlinear_features: List[str] = ['sd1', 'sd2', 'sd1_sd2_ratio', 'poincare_area', 'sampen_rri', 'dfa_alpha1_rri'],
                 # For nolds features on RRI series
                 nolds_sampen_emb_dim_rri: int = 2,
                 nolds_dfa_window_min_rri: int = 4,
                 nolds_dfa_window_max_rri_factor: float = 0.25, # Max window = factor * len(rri)
                 output_prefix_per_signal_col: bool = True, # If multiple signals processed for HRV (uncommon)
                 min_rri_count_for_hrv: int = 30): # Minimum number of RRIs to compute robust HRV
        super().__init__()
        self.signal_column_glob = signal_column_glob
        self.sampling_rate_hz = sampling_rate_hz
        self.peak_detection_method = peak_detection_method
        self.min_peak_height_factor = min_peak_height_factor
        self.min_peak_distance_s = min_peak_distance_s
        self.time_domain_features = time_domain_features
        self.freq_domain_features = freq_domain_features
        self.freq_bands_hrv_hz = freq_bands_hrv_hz
        self.nonlinear_features = nonlinear_features
        self.nolds_sampen_emb_dim_rri = nolds_sampen_emb_dim_rri
        self.nolds_dfa_window_min_rri = nolds_dfa_window_min_rri
        self.nolds_dfa_window_max_rri_factor = nolds_dfa_window_max_rri_factor
        self.output_prefix_per_signal_col = output_prefix_per_signal_col
        self.min_rri_count_for_hrv = min_rri_count_for_hrv

        if self.peak_detection_method == 'heartpy' and not has_heartpy:
            warn("'heartpy' peak detection selected but HeartPy library not found. Falling back to 'simple_scipy'.")
            self.peak_detection_method = 'simple_scipy'

    def _get_signal_columns(self, X: pd.DataFrame) -> List[str]:
        # Typically, HRV is derived from one primary signal (e.g., processed IR)
        # This might need to be more specific (e.g., user specifies THE signal column)
        if self.signal_column_glob == "proc_*": # Find first proc_ signal
            cols = [col for col in X.columns if col.startswith("proc_") and isinstance(X[col].iloc[0], (np.ndarray, list))]
            return cols[:1] if cols else [] # Return only the first one found, or empty
        else: # User specified glob
            cols = [col for col in X.columns if col.startswith(self.signal_column_glob.replace("*","")) and isinstance(X[col].iloc[0], (np.ndarray, list))]
            return cols[:1] if cols else []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        super().fit(X,y)
        self.input_signal_columns_ = self._get_signal_columns(X) # Should ideally be one specific column
        if not self.input_signal_columns_:
            warn(f"HeartRateVariabilityTransformer: No signal column found matching glob '{self.signal_column_glob}' for HRV.")

        temp_created_names = []
        # Assuming HRV features are global, not per input signal column if multiple were somehow processed.
        # If self.input_signal_columns_ is empty, this loop won't run, created_feature_names_ remains empty.
        signal_col_name_for_prefix = self.input_signal_columns_[0] if self.input_signal_columns_ else "hrv" # Fallback prefix
        
        prefix = f"{signal_col_name_for_prefix}_hrv_" if self.output_prefix_per_signal_col else "hrv_"

        for feat in self.time_domain_features: temp_created_names.append(f"{prefix}{feat}")
        for feat in self.freq_domain_features: temp_created_names.append(f"{prefix}{feat}")
        for band_name in self.freq_bands_hrv_hz.keys(): # Specific band powers if not covered by general freq_domain_features
            if f"ulf_power" not in self.freq_domain_features and band_name == 'ulf': temp_created_names.append(f"{prefix}ulf_power")
            if f"vlf_power" not in self.freq_domain_features and band_name == 'vlf': temp_created_names.append(f"{prefix}vlf_power")
        for feat in self.nonlinear_features: temp_created_names.append(f"{prefix}{feat}")
        
        self.created_feature_names_ = sorted(list(set(temp_created_names)))
        self.non_signal_columns_ = [col for col in X.columns if col not in self.input_signal_columns_]
        return self

    def _detect_peaks_simple_scipy(self, signal_array: np.ndarray) -> np.ndarray:
        if len(signal_array) == 0: return np.array([])
        min_h = np.min(signal_array) + self.min_peak_height_factor * (np.max(signal_array) - np.min(signal_array))
        min_dist_samples = int(self.min_peak_distance_s * self.sampling_rate_hz)
        if min_dist_samples < 1: min_dist_samples = 1
        
        peaks, _ = scipy_signal.find_peaks(signal_array, height=min_h, distance=min_dist_samples)
        return peaks

    def _detect_peaks_heartpy(self, signal_array: np.ndarray) -> np.ndarray:
        if not has_heartpy: return self._detect_peaks_simple_scipy(signal_array) # Fallback
        try:
            # HeartPy's process function can be resource-intensive; might use parts of it.
            # For peak detection: hp.peakdetection.detect_peaks
            working_data, measures = hp.process(signal_array, self.sampling_rate_hz, report_time=False, calc_freq=False)
            return working_data['peaklist'] # Returns indices of peaks
        except Exception as e:
            logging.warning(f"HeartPy peak detection failed: {e}. Falling back to simple_scipy.")
            return self._detect_peaks_simple_scipy(signal_array)

    def _calculate_hrv_features(self, rri_ms: np.ndarray, signal_col_name_for_prefix: str) -> Dict[str, float]:
        features = {}
        num_rri = len(rri_ms)
        prefix = f"{signal_col_name_for_prefix}_hrv_" if self.output_prefix_per_signal_col else "hrv_"

        # Pre-fill all expected HRV features with NaN or 0 to ensure consistent output columns
        for feat_name_template in self.created_feature_names_:
            if feat_name_template.startswith(prefix):
                features[feat_name_template] = np.nan

        if num_rri < self.min_rri_count_for_hrv:
            logging.debug(f"Not enough RRIs ({num_rri}) to calculate robust HRV. Min required: {self.min_rri_count_for_hrv}.")
            return features # Return dict with NaNs

        # Time-domain features
        if 'mean_rr' in self.time_domain_features: features[f"{prefix}mean_rr"] = np.mean(rri_ms)
        if 'sdnn' in self.time_domain_features: features[f"{prefix}sdnn"] = np.std(rri_ms)
        if 'hr_mean' in self.time_domain_features: features[f"{prefix}hr_mean"] = 60000.0 / np.mean(rri_ms) if np.mean(rri_ms) > 0 else np.nan
        if 'hr_std' in self.time_domain_features:
             hr_series_bpm = 60000.0 / rri_ms[rri_ms > 0]
             features[f"{prefix}hr_std"] = np.std(hr_series_bpm) if len(hr_series_bpm) > 1 else np.nan

        diff_rri = np.diff(rri_ms)
        if len(diff_rri) > 0 :
            if 'rmssd' in self.time_domain_features: features[f"{prefix}rmssd"] = np.sqrt(np.mean(diff_rri**2))
            if 'pnn50' in self.time_domain_features: features[f"{prefix}pnn50"] = 100.0 * np.sum(np.abs(diff_rri) > 50) / len(diff_rri)
        
        # Frequency-domain features (Lomb-Scargle on unevenly sampled RRI times)
        # Create timestamps for RRIs
        rri_times_s = np.cumsum(rri_ms) / 1000.0
        rri_times_s = rri_times_s - rri_times_s[0] # Start from 0

        # Interpolate RRIs to an evenly sampled series for FFT, or use Lomb-Scargle
        # Using Lomb-Scargle is generally preferred for uneven RRIs
        # Frequencies for Lomb-Scargle (up to Nyquist of about 0.5 Hz for HR of 60bpm)
        # Max freq for HRV is ~0.4-0.5 Hz.
        # Create a frequency grid
        highest_hrv_freq = max(b[1] for b in self.freq_bands_hrv_hz.values()) # e.g. 0.4 Hz
        min_freq_res = 0.001 # Hz
        # Frequencies for periodogram: from min_freq_res up to highest_hrv_freq (or Nyquist if lower)
        # The effective Nyquist for RRI series is ~0.5 / mean_rri_s
        mean_rri_s = np.mean(rri_ms) / 1000.0
        nyquist_rri = 0.5 / mean_rri_s if mean_rri_s > 0 else highest_hrv_freq
        
        f_eval = np.arange(min_freq_res, min(highest_hrv_freq + min_freq_res, nyquist_rri), min_freq_res)

        if len(f_eval) > 0 and len(rri_times_s) > 1 and np.std(rri_ms) > 1e-3 : # Need variance in RRIs
            try:
                # Detrend RRI series before periodogram (subtract mean)
                rri_ms_detrended = rri_ms - np.mean(rri_ms)
                power = scipy_signal.lombscargle(rri_times_s, rri_ms_detrended, f_eval, normalize=True) # Power units (ms^2)

                total_hrv_power_calc = 0.0
                for band_name, (low_f, high_f) in self.freq_bands_hrv_hz.items():
                    band_indices = np.where((f_eval >= low_f) & (f_eval < high_f))[0]
                    if len(band_indices) > 0:
                        band_power = np.trapz(power[band_indices], f_eval[band_indices]) # Integrate power
                        features[f"{prefix}{band_name}_power"] = band_power
                        if band_name in ['vlf', 'lf', 'hf']: # Standard bands for total power
                            total_hrv_power_calc += band_power
                
                if 'total_power_hrv' in self.freq_domain_features: features[f"{prefix}total_power_hrv"] = total_hrv_power_calc
                lf_power = features.get(f"{prefix}lf_power", 0.0)
                hf_power = features.get(f"{prefix}hf_power", 0.0)
                if 'lf_hf_ratio' in self.freq_domain_features: features[f"{prefix}lf_hf_ratio"] = self._safe_division(lf_power, hf_power)
            except Exception as e:
                logging.warning(f"HRV frequency domain calculation failed: {e}")
                # NaNs are already set

        # Non-linear features (Poincare, SampEn on RRI, DFA on RRI)
        if len(diff_rri) > 1: # Need at least 2 differences for Poincare
            if 'sd1' in self.nonlinear_features:
                # SD1 = sqrt(0.5 * std(X_n - X_{n-1})^2)
                features[f"{prefix}sd1"] = np.std(diff_rri / np.sqrt(2))
            if 'sd2' in self.nonlinear_features:
                # SD2 = sqrt(0.5 * std(X_n + X_{n-1} - 2*mean_X)^2) but simpler: std of (x_i + x_{i+1})/2
                # More standard: SD2 related to long-term variability
                # SD2^2 = Var( (X_n + X_{n+1})/2 ) - (SDNN^2)/2 -- complex
                # Alternative simplified: sqrt(2*SDNN^2 - 0.5*Var(X_n - X_{n-1}))
                # Using common approximation: SD2 = sqrt(2 * std(rri_ms)^2 - SD1^2)
                sd1_val = features.get(f"{prefix}sd1", 0)
                sdnn_val = features.get(f"{prefix}sdnn", 0)
                sd2_squared = 2 * sdnn_val**2 - sd1_val**2
                features[f"{prefix}sd2"] = np.sqrt(sd2_squared) if sd2_squared > 0 else np.nan
            
            sd1 = features.get(f"{prefix}sd1", np.nan)
            sd2 = features.get(f"{prefix}sd2", np.nan)
            if 'sd1_sd2_ratio' in self.nonlinear_features: features[f"{prefix}sd1_sd2_ratio"] = self._safe_division(sd1, sd2)
            if 'poincare_area' in self.nonlinear_features: features[f"{prefix}poincare_area"] = np.pi * sd1 * sd2

        if has_nolds and len(rri_ms) > 10 : # Min length for nolds on RRI
            if 'sampen_rri' in self.nonlinear_features:
                try:
                    sampen_tol_rri = 0.2 * np.std(rri_ms)
                    if sampen_tol_rri < 1e-3 : sampen_tol_rri = np.mean(rri_ms)*0.05 # 5% of mean RRI if std is tiny
                    features[f"{prefix}sampen_rri"] = nolds.sampen(rri_ms, emb_dim=self.nolds_sampen_emb_dim_rri, tolerance=sampen_tol_rri)
                except Exception as e: logging.warning(f"Nolds sampen for RRI failed: {e}") # NaN already set
            
            if 'dfa_alpha1_rri' in self.nonlinear_features: # Short-term DFA exponent
                try:
                    # nolds.dfa requires nvals argument (list of box sizes)
                    # Create nvals from min_window to max_window
                    min_n = self.nolds_dfa_window_min_rri
                    max_n = int(len(rri_ms) * self.nolds_dfa_window_max_rri_factor)
                    max_n = max(min_n + 5, max_n) # Ensure max_n is sufficiently larger than min_n
                    if max_n > len(rri_ms) / 2 : max_n = int(len(rri_ms) / 2) # Cap max_n
                    
                    if max_n > min_n and len(rri_ms) > max_n : # Ensure valid range and enough data
                        n_vals = np.array(np.logspace(np.log10(min_n), np.log10(max_n), num=10), dtype=int)
                        n_vals = np.unique(n_vals[n_vals < len(rri_ms)/2]) # Ensure unique and valid box sizes
                        if len(n_vals)>2:
                            # nolds.dfa can return an array of alphas if multiple orders are fit.
                            # We usually want alpha1 (for order=1, short-term fluctuations)
                            # This might need more careful parsing if nolds.dfa is changed
                            dfa_exponent = nolds.dfa(rri_ms, nvals=n_vals, order=1) # Order 1 for alpha1
                            features[f"{prefix}dfa_alpha1_rri"] = dfa_exponent
                        # else: not enough distinct n_vals
                    # else: not enough data for specified n_vals range
                except Exception as e: logging.warning(f"Nolds dfa for RRI failed: {e}") # NaN already set
        return features

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        super().transform(X)
        if not self.input_signal_columns_: # No signal column identified
            return X[self.non_signal_columns_] if self.non_signal_columns_ else X

        signal_col_for_hrv = self.input_signal_columns_[0] # Use the first (and ideally only) signal column

        all_row_features = []
        for _, row in X.iterrows():
            row_hrv_features = {}
            # Keep non-signal columns
            for col_name in self.non_signal_columns_:
                row_hrv_features[col_name] = row[col_name]
            
            signal_array = row.get(signal_col_for_hrv)
            
            # Pre-fill all HRV features with NaN for this row
            # This ensures DataFrame consistency if HRV cannot be computed
            hrv_prefix_current = f"{signal_col_for_hrv}_hrv_" if self.output_prefix_per_signal_col else "hrv_"
            for feat_name_template in self.created_feature_names_:
                if feat_name_template.startswith(hrv_prefix_current):
                     row_hrv_features[feat_name_template] = np.nan


            if signal_array is None or not isinstance(signal_array, (np.ndarray, list)) or len(signal_array) < int(self.sampling_rate_hz * self.min_peak_distance_s * 5) : # Min length for a few peaks
                logging.debug(f"Signal '{signal_col_for_hrv}' invalid or too short for HRV. Length: {len(signal_array if signal_array is not None else [])}")
                all_row_features.append(row_hrv_features) # Add row with NaNs for HRV features
                continue

            signal_array = np.asarray(signal_array, dtype=float)

            if self.peak_detection_method == 'simple_scipy':
                peak_indices = self._detect_peaks_simple_scipy(signal_array)
            elif self.peak_detection_method == 'heartpy':
                peak_indices = self._detect_peaks_heartpy(signal_array)
            else: # Should not happen if init validates
                peak_indices = np.array([])
            
            if len(peak_indices) < 5: # Need at least a few peaks for meaningful RRIs
                logging.debug(f"Too few peaks ({len(peak_indices)}) detected in '{signal_col_for_hrv}' for HRV.")
                all_row_features.append(row_hrv_features) # Add row with NaNs for HRV features
                continue

            # Calculate RR intervals in milliseconds
            rri_samples = np.diff(peak_indices)
            rri_ms = (rri_samples / self.sampling_rate_hz) * 1000.0
            
            # Basic RRI cleaning (remove extreme outliers, e.g., physiologically implausible)
            rri_ms_cleaned = rri_ms[(rri_ms > 250) & (rri_ms < 2000)] # Typical physiological range for RRIs
            
            if len(rri_ms_cleaned) >= self.min_rri_count_for_hrv:
                calculated_hrv_feats = self._calculate_hrv_features(rri_ms_cleaned, signal_col_for_hrv)
                row_hrv_features.update(calculated_hrv_feats) # Update the pre-filled NaNs with actual values
            else:
                 logging.debug(f"Not enough cleaned RRIs ({len(rri_ms_cleaned)}) for HRV. Min: {self.min_rri_count_for_hrv}")
                 # NaNs already set for HRV features for this row

            all_row_features.append(row_hrv_features)

        if not all_row_features:
            logging.warning("HeartRateVariabilityTransformer: No features extracted.")
            cols_to_return = self.non_signal_columns_ + self.created_feature_names_
            return pd.DataFrame(columns=list(dict.fromkeys(cols_to_return)))

        features_df = pd.DataFrame(all_row_features)
        # Ensure all expected columns exist, even if all were NaN
        for expected_feature_name in self.created_feature_names_:
            if expected_feature_name not in features_df.columns:
                features_df[expected_feature_name] = np.nan
        
        final_columns = self.non_signal_columns_ + self.created_feature_names_
        final_columns = [col for col in final_columns if col in features_df.columns]
        
        # Impute NaNs with a strategy (e.g., median of the column, or 0)
        # For now, using 0.0, but a dedicated imputer in the pipeline is better.
        return features_df[list(dict.fromkeys(final_columns))].fillna(0.0)


class MultiSpectralFeatureTransformer(BaseSignalTransformer):
    """
    Creates advanced ratiometric and physics-inspired features across multiple
    wavelength channels. Assumes input DataFrame X contains columns with
    statistical features (mean, std, AC/DC, etc.) for different wavelengths,
    prefixed by wavelength name (e.g., 'proc_red_mean', 'proc_ir_ac_amplitude').
    """
    def __init__(self,
                 wavelength_channel_prefixes: List[str] = ['proc_red', 'proc_ir', 'proc_green', 'proc_blue'], # Prefixes from PPGPreprocessor
                 # Suffixes identify type of feature extracted per channel (e.g., by another transformer)
                 mean_suffix: str = '_mean', # e.g., proc_red_mean
                 std_suffix: str = '_std',
                 ac_amplitude_suffix: str = '_ac_amp', # Amplitude of pulsatile component
                 dc_offset_suffix: str = '_dc_offset', # Baseline non-pulsatile component
                 # Add other relevant suffixes like _skew, _kurt, _p2p_amp, etc.
                 biomarkers_of_interest: List[str] = ['spo2', 'glucose', 'total_cholesterol', 'hemoglobin'],
                 custom_ratios: Optional[List[Tuple[str, str, str]]] = None, # (num_feat_key, den_feat_key, ratio_name)
                 output_prefix: str = "msf_"): # MultiSpectral Feature
        super().__init__()
        self.wavelength_channel_prefixes = wavelength_channel_prefixes
        self.mean_suffix = mean_suffix
        self.std_suffix = std_suffix
        self.ac_amplitude_suffix = ac_amplitude_suffix
        self.dc_offset_suffix = dc_offset_suffix
        self.biomarkers_of_interest = [b.lower() for b in biomarkers_of_interest]
        self.custom_ratios = custom_ratios if custom_ratios else []
        self.output_prefix = output_prefix
        self.available_wavelength_features_ = {} # To be populated in fit

    def _get_available_wavelength_features(self, X: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        # Discovers available features per wavelength prefix
        # Returns: {'red': {'mean': 'proc_red_mean', 'std': 'proc_red_std', ...}, 'ir': {...}}
        available = {}
        all_suffixes = {
            'mean': self.mean_suffix, 'std': self.std_suffix,
            'ac': self.ac_amplitude_suffix, 'dc': self.dc_offset_suffix
        }
        for wl_prefix_full in self.wavelength_channel_prefixes: # e.g., 'proc_red'
            # Infer short wavelength name like 'red' from 'proc_red'
            short_wl_name = wl_prefix_full.split('_')[-1] if '_' in wl_prefix_full else wl_prefix_full

            available[short_wl_name] = {}
            for feat_type, suffix in all_suffixes.items():
                col_name = f"{wl_prefix_full}{suffix}" # e.g., proc_red_mean
                if col_name in X.columns:
                    available[short_wl_name][feat_type] = col_name
        return available

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        super().fit(X, y)
        self.available_wavelength_features_ = self._get_available_wavelength_features(X)
        if not self.available_wavelength_features_ or not any(self.available_wavelength_features_.values()):
            warn("MultiSpectralFeatureTransformer: No wavelength-specific features found. Transformer may produce no output.")

        temp_created_names = []
        # Standard ratios and NDIs
        wl_names = list(self.available_wavelength_features_.keys())
        for i in range(len(wl_names)):
            for j in range(i + 1, len(wl_names)):
                wl1, wl2 = wl_names[i], wl_names[j]
                # Check if 'mean' features exist for these wavelengths
                if 'mean' in self.available_wavelength_features_.get(wl1, {}) and \
                   'mean' in self.available_wavelength_features_.get(wl2, {}):
                    temp_created_names.append(f"{self.output_prefix}{wl1}_{wl2}_mean_ratio")
                    temp_created_names.append(f"{self.output_prefix}log_{wl1}_{wl2}_mean_ratio")
                    temp_created_names.append(f"{self.output_prefix}{wl1}_{wl2}_ndi_mean") # NDI from mean values

                # Ratios of AC/DC if available (more common for SpO2-like calculations)
                if 'ac' in self.available_wavelength_features_.get(wl1, {}) and \
                   'dc' in self.available_wavelength_features_.get(wl1, {}) and \
                   'ac' in self.available_wavelength_features_.get(wl2, {}) and \
                   'dc' in self.available_wavelength_features_.get(wl2, {}):
                    temp_created_names.append(f"{self.output_prefix}{wl1}_{wl2}_ac_dc_ratio_of_ratios")


        # Biomarker-specific features
        if 'spo2' in self.biomarkers_of_interest:
            # Standard R-ratio (Red AC/DC) / (IR AC/DC)
            if 'red' in wl_names and 'ir' in wl_names and \
               all(k in self.available_wavelength_features_['red'] for k in ['ac', 'dc']) and \
               all(k in self.available_wavelength_features_['ir'] for k in ['ac', 'dc']):
                temp_created_names.append(f"{self.output_prefix}r_ratio_spo2")
                temp_created_names.append(f"{self.output_prefix}empirical_spo2_from_r_ratio")

        if 'glucose' in self.biomarkers_of_interest:
            # Example: Ratio of NIR STDs if multiple NIR bands are somehow identified/prefixed
            # This requires more specific prefix handling than current generic `wavelength_channel_prefixes`
            # For now, use a placeholder or specific known NIR channels if they exist in wl_names
            nir_channels = [wl for wl in wl_names if 'nir' in wl.lower() or wl in ['940nm', '1550nm']] # Heuristic
            if len(nir_channels) >= 2:
                nir1, nir2 = nir_channels[0], nir_channels[1]
                if 'std' in self.available_wavelength_features_.get(nir1, {}) and \
                   'std' in self.available_wavelength_features_.get(nir2, {}):
                     temp_created_names.append(f"{self.output_prefix}{nir1}_{nir2}_std_ratio_glucose")

        if 'hemoglobin' in self.biomarkers_of_interest:
            if 'green' in wl_names and 'red' in wl_names and \
               'mean' in self.available_wavelength_features_.get('green',{}) and \
               'mean' in self.available_wavelength_features_.get('red',{}):
                temp_created_names.append(f"{self.output_prefix}green_red_mean_ratio_hb")

        if 'total_cholesterol' in self.biomarkers_of_interest or 'lipids' in self.biomarkers_of_interest:
            # Scattering related, e.g., ratio of STDs at different wavelengths
            if 'red' in wl_names and 'ir' in wl_names and \
               'std' in self.available_wavelength_features_.get('red',{}) and \
               'std' in self.available_wavelength_features_.get('ir',{}):
                temp_created_names.append(f"{self.output_prefix}red_ir_std_ratio_lipids")

        for _, _, ratio_name in self.custom_ratios:
            temp_created_names.append(f"{self.output_prefix}{ratio_name}")

        self.created_feature_names_ = sorted(list(set(temp_created_names)))
        # Identify non-feature columns (original columns that are not raw signals or their stats)
        # This is tricky as this transformer operates on features from previous transformers.
        # For simplicity, assume all input columns to this transformer are feature columns.
        # Or, one could pass through columns not matching wavelength_channel_prefixes + suffixes.
        # For now, assume it consumes all input and produces new features.
        self.passthrough_columns_ = [col for col in X.columns if col not in self.get_all_used_input_feature_cols(X)]
        return self

    def get_all_used_input_feature_cols(self, X:pd.DataFrame) -> List[str]:
        """Helper to get all columns this transformer might use based on its config."""
        used_cols = set()
        temp_avail_feats = self._get_available_wavelength_features(X) # Check current X
        for wl_data in temp_avail_feats.values():
            used_cols.update(wl_data.values())
        for num_feat, den_feat, _ in self.custom_ratios:
            used_cols.add(num_feat)
            used_cols.add(den_feat)
        return list(used_cols)


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        super().transform(X) # Basic checks
        
        # If available_wavelength_features_ was empty from fit, this will do little.
        # Re-check on current X in case columns changed, though fit should establish structure.
        current_avail_wl_feats = self._get_available_wavelength_features(X)
        if not current_avail_wl_feats or not any(current_avail_wl_feats.values()):
            # Return passthrough columns if any, or X itself
            return X[self.passthrough_columns_].copy() if self.passthrough_columns_ else X.copy()

        output_X = pd.DataFrame(index=X.index)

        # Standard Ratios and NDIs
        wl_names = list(current_avail_wl_feats.keys())
        for i in range(len(wl_names)):
            for j in range(i + 1, len(wl_names)):
                wl1, wl2 = wl_names[i], wl_names[j]
                
                mean1_col = current_avail_wl_feats.get(wl1, {}).get('mean')
                mean2_col = current_avail_wl_feats.get(wl2, {}).get('mean')
                if mean1_col and mean2_col and mean1_col in X.columns and mean2_col in X.columns:
                    num = X[mean1_col]
                    den = X[mean2_col]
                    output_X[f"{self.output_prefix}{wl1}_{wl2}_mean_ratio"] = self._safe_division(num, den)
                    output_X[f"{self.output_prefix}log_{wl1}_{wl2}_mean_ratio"] = np.log1p(np.abs(output_X[f"{self.output_prefix}{wl1}_{wl2}_mean_ratio"])) # log(1+|x|)
                    output_X[f"{self.output_prefix}{wl1}_{wl2}_ndi_mean"] = self._safe_division(num - den, num + den)

                ac1_col = current_avail_wl_feats.get(wl1, {}).get('ac')
                dc1_col = current_avail_wl_feats.get(wl1, {}).get('dc')
                ac2_col = current_avail_wl_feats.get(wl2, {}).get('ac')
                dc2_col = current_avail_wl_feats.get(wl2, {}).get('dc')
                if all(c in X.columns for c in [ac1_col, dc1_col, ac2_col, dc2_col] if c is not None):
                    ratio1 = self._safe_division(X[ac1_col], X[dc1_col])
                    ratio2 = self._safe_division(X[ac2_col], X[dc2_col])
                    output_X[f"{self.output_prefix}{wl1}_{wl2}_ac_dc_ratio_of_ratios"] = self._safe_division(ratio1, ratio2)

        # Biomarker-specific features
        if 'spo2' in self.biomarkers_of_interest:
            red_ac_col = current_avail_wl_feats.get('red', {}).get('ac')
            red_dc_col = current_avail_wl_feats.get('red', {}).get('dc')
            ir_ac_col = current_avail_wl_feats.get('ir', {}).get('ac') # Assuming 'ir' is a key like '805nm' or '940nm'
            ir_dc_col = current_avail_wl_feats.get('ir', {}).get('dc')
            # A more robust way: find specific NIR wavelengths like 805nm or 940nm from wl_names
            # For this example, assume 'ir' is one of the `wavelength_channel_prefixes` simplified name
            
            if all(c in X.columns for c in [red_ac_col, red_dc_col, ir_ac_col, ir_dc_col] if c is not None):
                r_ratio_num = self._safe_division(X[red_ac_col], X[red_dc_col])
                r_ratio_den = self._safe_division(X[ir_ac_col], X[ir_dc_col])
                r_ratio = self._safe_division(r_ratio_num, r_ratio_den)
                output_X[f"{self.output_prefix}r_ratio_spo2"] = r_ratio
                # Standard empirical formula (example: 110 - 25 * R, or 104 - 17*R, varies)
                # This should be calibrated. Using a common one.
                output_X[f"{self.output_prefix}empirical_spo2_from_r_ratio"] = np.clip(104.0 - 17.0 * r_ratio, 0.5, 1.0)


        if 'glucose' in self.biomarkers_of_interest:
            # Find NIR channels (e.g., >900nm), look for specific absorption/scattering changes.
            # Example: Difference in attenuation at two NIR wavelengths, one more glucose-sensitive.
            # This highly depends on the available wavelengths and their OPTICAL_CONSTANTS.
            # Let's assume 'nir_long' and 'nir_short' are two such identified feature sets.
            # This part is highly conceptual without specific NIR channel features defined.
            # If 1550nm and 1370nm features (e.g., mean attenuation) are available:
            atten_1550_col = current_avail_wl_feats.get('1550nm', {}).get('mean') # Assuming 'mean' is a proxy for attenuation
            atten_1370_col = current_avail_wl_feats.get('1370nm', {}).get('mean')
            if atten_1550_col and atten_1370_col and atten_1550_col in X.columns and atten_1370_col in X.columns:
                # (Attenuation at peak - Attenuation at reference) / Attenuation at reference
                # This is a very simplified differential absorption concept.
                output_X[f"{self.output_prefix}nir_diff_abs_glucose_proxy"] = \
                    self._safe_division(X[atten_1550_col] - X[atten_1370_col], X[atten_1370_col])

        if 'hemoglobin' in self.biomarkers_of_interest:
            green_mean_col = current_avail_wl_feats.get('green', {}).get('mean')
            red_mean_col = current_avail_wl_feats.get('red', {}).get('mean')
            if green_mean_col and red_mean_col and green_mean_col in X.columns and red_mean_col in X.columns:
                # Hb absorbs strongly in green. Red is less absorbed.
                output_X[f"{self.output_prefix}green_red_mean_ratio_hb"] = self._safe_division(X[green_mean_col], X[red_mean_col])

        if 'total_cholesterol' in self.biomarkers_of_interest or 'lipids' in self.biomarkers_of_interest:
            # Lipids increase scattering. Changes in signal STD or specific NIR ratios might be indicative.
            # Example: std at a shorter wavelength (more scattering sensitive) vs. longer NIR.
            red_std_col = current_avail_wl_feats.get('red', {}).get('std')
            ir_std_col = current_avail_wl_feats.get('ir', {}).get('std') # e.g. 940nm
            if red_std_col and ir_std_col and red_std_col in X.columns and ir_std_col in X.columns:
                 output_X[f"{self.output_prefix}red_ir_std_ratio_lipids"] = self._safe_division(X[red_std_col], X[ir_std_col])

        # Custom Ratios
        for num_feat_key, den_feat_key, ratio_name in self.custom_ratios:
            if num_feat_key in X.columns and den_feat_key in X.columns:
                output_X[f"{self.output_prefix}{ratio_name}"] = self._safe_division(X[num_feat_key], X[den_feat_key])
            else:
                output_X[f"{self.output_prefix}{ratio_name}"] = 0.0 # Or NaN

        # Concatenate with passthrough columns
        # Passthrough columns are those from original X not used by this transformer.
        # This logic might need refinement if X itself is the output of another transformer.
        # For now, assume X contains all original features + features from previous steps.
        passthrough_df = X[self.passthrough_columns_].copy()
        
        final_df = pd.concat([passthrough_df, output_X], axis=1)
        
        # Ensure all `created_feature_names_` are present in the output, fill with 0 if missing
        for feat_name in self.created_feature_names_:
            if feat_name not in final_df.columns:
                final_df[feat_name] = 0.0
        
        # Return only expected columns (passthrough + created) in a defined order
        expected_final_cols = self.passthrough_columns_ + self.created_feature_names_
        expected_final_cols = [col for col in expected_final_cols if col in final_df.columns] # Filter to existing
        return final_df[list(dict.fromkeys(expected_final_cols))].fillna(0.0)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if not self._is_fitted:
             warn("get_feature_names_out called before MultiSpectralFeatureTransformer fit. Returning empty list or input_features if provided.")
             return input_features if input_features is not None else []
        
        # If input_features are provided, they are the columns output by the previous step.
        # We need to identify which of these are passthrough.
        # This is simpler if `fit` correctly identifies `passthrough_columns_`.
        passthrough = self.passthrough_columns_
        if input_features is not None and not passthrough: # If passthrough not set by fit (e.g. fit on different cols)
            # Attempt to infer passthrough columns
            # This is complex; a robust way is to ensure `fit` always defines `passthrough_columns_`
            # based on the columns it was fitted on.
            # For simplicity, assume `self.passthrough_columns_` is correctly set by `fit`.
            pass

        return list(dict.fromkeys(self.passthrough_columns_ + self.created_feature_names_))

# ... (Continuing from the previous code block, including all Feature Extractors up to MultiSpectralFeatureTransformer) ...

# =============== DEEP LEARNING MODEL BUILDING BLOCKS (Finalized) ===============

class ResidualConv1DBlock(tf.keras.layers.Layer):
    """
    A Residual 1D Convolutional Block with optional Squeeze-and-Excitation.
    Implements a 'full pre-activation' style residual unit if use_batch_norm is True.
    """
    def __init__(self, filters: int, kernel_size: int, strides: int = 1,
                 activation: str = 'relu', dropout_rate: float = 0.1,
                 use_batch_norm: bool = True, use_squeeze_excitation: bool = False,
                 se_ratio: int = 4, l2_reg: float = 1e-4, dilation_rate: int = 1, **kwargs):
        super(ResidualConv1DBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation_name = activation # Store name for get_config
        self.activation = tf.keras.activations.get(activation) # Get actual function
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_squeeze_excitation = use_squeeze_excitation
        self.se_ratio = se_ratio
        self.l2_reg = l2_reg
        self.dilation_rate = dilation_rate

        # Convolutional Path
        self.bn1 = BatchNormalization() if use_batch_norm else tf.identity
        self.conv1 = Conv1D(filters, kernel_size, strides=strides, padding='same', dilation_rate=dilation_rate,
                            kernel_regularizer=l1_l2(l2=l2_reg), use_bias=not use_batch_norm)
        
        self.bn2 = BatchNormalization() if use_batch_norm else tf.identity
        self.conv2 = Conv1D(filters, kernel_size, strides=1, padding='same', dilation_rate=dilation_rate,
                            kernel_regularizer=l1_l2(l2=l2_reg), use_bias=not use_batch_norm)
        
        self.drop_out = Dropout(dropout_rate) if dropout_rate > 0 else tf.identity
        
        if self.use_squeeze_excitation:
            self.se_block_layers = [
                GlobalAveragePooling1D(keepdims=True),
                Dense(max(1, filters // self.se_ratio), activation='relu', kernel_regularizer=l1_l2(l2=self.l2_reg)),
                Dense(filters, activation='sigmoid', kernel_regularizer=l1_l2(l2=self.l2_reg)),
                Multiply()
            ]
        
        # Shortcut connection
        self.shortcut_conv = None # Will be defined in build if needed
        self.add_layer = Add()

    def build(self, input_shape):
        input_channels = input_shape[-1]
        if self.strides > 1 or input_channels != self.filters:
            self.shortcut_conv = Conv1D(self.filters, kernel_size=1, strides=self.strides, padding='same',
                                        kernel_regularizer=l1_l2(l2=self.l2_reg), name=f"{self.name}_shortcut_conv" if self.name else "shortcut_conv")
        super(ResidualConv1DBlock, self).build(input_shape)

    def call(self, inputs, training=False):
        # Full pre-activation style for the main path
        x = self.bn1(inputs, training=training)
        x = self.activation(x)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.drop_out(x, training=training)

        if self.use_squeeze_excitation:
            se_weights = x
            for layer in self.se_block_layers[:-1]: # All but Multiply
                se_weights = layer(se_weights, training=training if isinstance(layer, Dropout) else None)
            x = self.se_block_layers[-1]([x, se_weights]) # Multiply

        shortcut = inputs
        if self.shortcut_conv:
            shortcut = self.shortcut_conv(inputs) # Shortcut might also need pre-activation if it includes conv
        
        res_out = self.add_layer([x, shortcut])
        # No final activation here, typically done after the Add in resnet v1, or as pre-activation in next block for resnet v2
        # For simplicity of stacking, let's assume this block's output can be directly fed to next.
        return res_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters, "kernel_size": self.kernel_size, "strides": self.strides,
            "activation": self.activation_name, "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm, "use_squeeze_excitation": self.use_squeeze_excitation,
            "se_ratio": self.se_ratio, "l2_reg": self.l2_reg, "dilation_rate": self.dilation_rate
        })
        return config

class AttentionBlock(tf.keras.layers.Layer):
    """
    Multi-Head Attention block with Add & Norm.
    """
    def __init__(self, num_heads: int, key_dim: int, value_dim: Optional[int] = None,
                 ff_dim: Optional[int] = None, # Dimension of the feed-forward network, if used
                 dropout_rate: float = 0.1, use_bias: bool = True, l2_reg: float = 1e-4, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.l2_reg = l2_reg

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=self.value_dim,
                                      dropout=dropout_rate, use_bias=use_bias,
                                      kernel_regularizer=l1_l2(l2=l2_reg))
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.add1 = Add()
        
        if self.ff_dim:
            self.ffn = Sequential([
                Dense(ff_dim, activation=tf.nn.gelu, kernel_regularizer=l1_l2(l2=l2_reg)),
                Dropout(dropout_rate),
                Dense(key_dim * num_heads if value_dim is None else value_dim * num_heads, kernel_regularizer=l1_l2(l2=l2_reg)) # Project back to MHA output dim
            ], name=f"{self.name}_ffn" if self.name else "ffn")
            self.layernorm2 = LayerNormalization(epsilon=1e-6)
            self.add2 = Add()
        else:
            self.ffn = None

    def call(self, inputs, attention_mask=None, training=False):
        if isinstance(inputs, (list, tuple)):
            query, value, key = inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else inputs[1]
        else:
            query, value, key = inputs, inputs, inputs
        
        attn_output = self.mha(query=query, value=value, key=key,
                               attention_mask=attention_mask, training=training)
        x = self.add1([query, attn_output])
        x = self.layernorm1(x)
        
        if self.ffn:
            ffn_output = self.ffn(x, training=training)
            x = self.add2([x, ffn_output])
            x = self.layernorm2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads, "key_dim": self.key_dim, "value_dim": self.value_dim,
            "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias, "l2_reg": self.l2_reg
        })
        return config

# =============== SPECIALIZED DEEP LEARNING MODELS (Finalized) ===============

def build_spectral_attention_cnn(input_shape: Tuple, num_outputs: int, biomarker_name: str,
                                 config: Dict) -> Model:
    """
    Builds a CNN model with Residual Blocks and Attention for spectral feature data.
    Input shape is (num_features,) or (sequence_length, num_features_per_step).
    Config keys: num_filters_list, kernel_sizes_list, res_strides_list, res_dilation_rates,
                 res_activation, res_block_dropout, use_squeeze_excitation, se_ratio,
                 attention_heads, attention_key_dim, attention_ff_dim, attention_dropout,
                 global_pooling_type ('avg', 'max', 'avgmax'),
                 final_dense_units, dropout_final_dense, dense_activation,
                 l2_reg_cnn, l2_reg_dense, output_activation_type.
    """
    inputs = Input(shape=input_shape, name=f"{biomarker_name}_input_features")
    x = inputs

    if len(input_shape) == 1: # Flat feature vector
        x = Reshape((input_shape[0], 1))(x) # Reshape to (features, 1) for Conv1D
    
    num_res_blocks = len(config['num_filters_list'])
    for i in range(num_res_blocks):
        filters = config['num_filters_list'][i]
        kernel_size = config['kernel_sizes_list'][i]
        strides = config['res_strides_list'][i] if i < len(config.get('res_strides_list',[])) else 1
        dilation = config['res_dilation_rates'][i] if i < len(config.get('res_dilation_rates',[])) else 1
        
        x = ResidualConv1DBlock(filters=filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation,
                                activation=config['res_activation'], dropout_rate=config['res_block_dropout'],
                                use_squeeze_excitation=config['use_squeeze_excitation'], se_ratio=config['se_ratio'],
                                l2_reg=config['l2_reg_cnn'], name=f"{biomarker_name}_res_conv_block_{i+1}")(x)
    
    if config.get('attention_heads', 0) > 0:
        x = AttentionBlock(num_heads=config['attention_heads'], key_dim=config['attention_key_dim'],
                           ff_dim=config.get('attention_ff_dim'), dropout_rate=config['attention_dropout'],
                           l2_reg=config['l2_reg_cnn'], name=f"{biomarker_name}_cnn_attention")(x)

    if config['global_pooling_type'] == 'avg':
        x = GlobalAveragePooling1D(name=f"{biomarker_name}_global_pool")(x)
    elif config['global_pooling_type'] == 'max':
        x = GlobalMaxPooling1D(name=f"{biomarker_name}_global_pool")(x)
    elif config['global_pooling_type'] == 'avgmax':
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        x = Concatenate(name=f"{biomarker_name}_global_pool")([avg_pool, max_pool])
    else: # No pooling or Flatten if sequence dim is 1
        x = Flatten()(x) if len(x.shape) > 2 else x

    for units in config['final_dense_units']:
        x = Dense(units, kernel_regularizer=l1_l2(l2=config['l2_reg_dense']), 
                  name=f"{biomarker_name}_dense_{units}")(x)
        x = BatchNormalization(name=f"{biomarker_name}_bn_dense_{units}")(x)
        x = Activation(config['dense_activation'])(x)
        x = Dropout(config['dropout_final_dense'], name=f"{biomarker_name}_drop_dense_{units}")(x)
    
    raw_output = Dense(num_outputs, activation='linear', name=f"{biomarker_name}_raw_output_logits")(x)
    constrained_output = PhysiologicalConstraintLayer(biomarker_name=biomarker_name,
                                                     activation_type=config['output_activation_type'],
                                                     name=f"{biomarker_name}_constrained_output")(raw_output)
    
    model = Model(inputs=inputs, outputs=constrained_output, name=f"{biomarker_name}_SpectralAttentionCNN")
    return model


def build_temporal_attention_lstm(input_shape: Tuple, num_outputs: int, biomarker_name: str,
                                  config: Dict) -> Model:
    """
    Builds an LSTM/GRU model with optional Attention for temporal sequence data.
    Input shape is (sequence_length, num_features_per_step).
    Config keys: rnn_type ('lstm', 'gru'), rnn_units_list, bidirectional, rnn_activation,
                 rnn_dropout, recurrent_dropout, attention_after_rnn, attention_heads,
                 attention_key_dim, attention_ff_dim, attention_dropout,
                 global_pooling_type_rnn ('avg', 'max', 'last', 'attn_pool'),
                 final_dense_units, dropout_final_dense, dense_activation,
                 l2_reg_rnn, l2_reg_dense, output_activation_type.
    """
    if len(input_shape) != 2:
        raise ValueError("Input shape for TemporalAttentionLSTM must be (sequence_length, num_features_per_step).")

    inputs = Input(shape=input_shape, name=f"{biomarker_name}_input_sequence")
    x = inputs
    
    RNNLayer = LSTM if config['rnn_type'].lower() == 'lstm' else GRU
    
    for i, units in enumerate(config['rnn_units_list']):
        is_last_rnn = (i == len(config['rnn_units_list']) - 1)
        # Return sequences if not last RNN, or if attention/further RNNs follow
        return_sequences = (not is_last_rnn) or (config['attention_after_rnn']) or (len(config['rnn_units_list']) > i + 1)
        
        layer_name = f"{biomarker_name}_{config['rnn_type']}_{i+1}"
        rnn_layer_instance = RNNLayer(units, return_sequences=return_sequences, activation=config['rnn_activation'],
                                   dropout=config['rnn_dropout'], recurrent_dropout=config['recurrent_dropout'],
                                   kernel_regularizer=l1_l2(l2=config['l2_reg_rnn']),
                                   recurrent_regularizer=l1_l2(l2=config['l2_reg_rnn']),
                                   name=layer_name)
        if config['bidirectional']:
            x = Bidirectional(rnn_layer_instance, name=f"bi_{layer_name}")(x)
        else:
            x = rnn_layer_instance(x)
        x = BatchNormalization(name=f"{biomarker_name}_bn_after_{layer_name}")(x)

    if config['attention_after_rnn'] and config.get('attention_heads',0) > 0:
        if not return_sequences: # Should be true if attention_after_rnn
             raise ValueError("Last RNN must return_sequences=True if attention_after_rnn is True.")
        x = AttentionBlock(num_heads=config['attention_heads'], key_dim=config['attention_key_dim'],
                           ff_dim=config.get('attention_ff_dim'), dropout_rate=config['attention_dropout'],
                           l2_reg=config['l2_reg_rnn'], name=f"{biomarker_name}_rnn_attention")(x)
    
    # Pooling / Selecting from sequence
    if x.shape[1] is not None and x.shape[1] > 1: # If still a sequence
        if config['global_pooling_type_rnn'] == 'avg': x = GlobalAveragePooling1D()(x)
        elif config['global_pooling_type_rnn'] == 'max': x = GlobalMaxPooling1D()(x)
        elif config['global_pooling_type_rnn'] == 'last': x = x[:, -1, :] # Take last time step output
        elif config['global_pooling_type_rnn'] == 'attn_pool' and config.get('attention_heads',0) > 0 : # If attention was applied, x is already pooled implicitly or ready
             x = GlobalAveragePooling1D(name=f"{biomarker_name}_rnn_final_pool")(x) # Ensure flat
        else: # Default to flatten or last step if not specified and still sequence
            x = Flatten()(x) if len(x.shape) > 2 else x


    for units in config['final_dense_units']:
        x = Dense(units, kernel_regularizer=l1_l2(l2=config['l2_reg_dense']),
                  name=f"{biomarker_name}_dense_{units}")(x)
        x = BatchNormalization(name=f"{biomarker_name}_bn_dense_{units}")(x)
        x = Activation(config['dense_activation'])(x)
        x = Dropout(config['dropout_final_dense'], name=f"{biomarker_name}_drop_dense_{units}")(x)
        
    raw_output = Dense(num_outputs, activation='linear', name=f"{biomarker_name}_raw_output_logits")(x)
    constrained_output = PhysiologicalConstraintLayer(biomarker_name=biomarker_name,
                                                     activation_type=config['output_activation_type'],
                                                     name=f"{biomarker_name}_constrained_output")(raw_output)
    
    model = Model(inputs=inputs, outputs=constrained_output, name=f"{biomarker_name}_TemporalAttentionRNN")
    return model


def build_hybrid_cnn_lstm_transformer(input_shape: Tuple, num_outputs: int, biomarker_name: str,
                                      config: Dict) -> Model:
    """
    Builds a Hybrid model: CNN -> LSTM -> Transformer Encoder blocks.
    Config keys: cnn_num_filters_list, cnn_kernel_sizes_list, cnn_strides_list, cnn_dilation_rates,
                 cnn_activation, cnn_dropout, use_cnn_se, cnn_se_ratio,
                 rnn_type, rnn_units_list, rnn_bidirectional, rnn_activation, rnn_dropout, rec_dropout_rnn,
                 transformer_num_blocks, transformer_num_heads, transformer_key_dim, transformer_value_dim_mult (of key_dim),
                 transformer_ff_dim_mult (of model_dim), transformer_dropout,
                 hybrid_projection_dim (dim before transformer, if 0, use rnn_output_dim),
                 global_pooling_type_hybrid,
                 final_dense_units, dropout_final_dense, dense_activation,
                 l2_reg, output_activation_type.
    """
    inputs = Input(shape=input_shape, name=f"{biomarker_name}_hybrid_input")
    x = inputs

    if len(input_shape) == 1:
        x = Reshape((input_shape[0], 1))(x)
    
    # 1. CNN Backbone
    num_cnn_blocks = len(config['cnn_num_filters_list'])
    for i in range(num_cnn_blocks):
        filters = config['cnn_num_filters_list'][i]
        kernel_size = config['cnn_kernel_sizes_list'][i]
        strides = config['cnn_strides_list'][i] if i < len(config.get('cnn_strides_list',[])) else 1
        dilation = config['cnn_dilation_rates'][i] if i < len(config.get('cnn_dilation_rates',[])) else 1
        x = ResidualConv1DBlock(filters, kernel_size, strides=strides, dilation_rate=dilation,
                                activation=config['cnn_activation'], dropout_rate=config['cnn_dropout'],
                                use_squeeze_excitation=config['use_cnn_se'], se_ratio=config.get('cnn_se_ratio',4),
                                l2_reg=config['l2_reg'], name=f"{biomarker_name}_hybrid_resconv_{i}")(x)

    # 2. RNN Layers
    RNNLayer = LSTM if config['rnn_type'].lower() == 'lstm' else GRU
    for i, units in enumerate(config['rnn_units_list']):
        rnn_layer_instance = RNNLayer(units, return_sequences=True, activation=config['rnn_activation'], # Must return sequences for Transformer
                                   dropout=config['rnn_dropout'], recurrent_dropout=config['rec_dropout_rnn'],
                                   kernel_regularizer=l1_l2(l2=config['l2_reg']), name=f"{biomarker_name}_hybrid_{config['rnn_type']}_{i}")
        if config['rnn_bidirectional']:
            x = Bidirectional(rnn_layer_instance, name=f"{biomarker_name}_hybrid_bi_{config['rnn_type']}_{i}")(x)
        else:
            x = rnn_layer_instance(x)
        x = BatchNormalization(name=f"{biomarker_name}_hybrid_{config['rnn_type']}_bn_{i}")(x)

    # 3. Transformer Encoder Blocks
    transformer_model_dim = config.get('hybrid_projection_dim', 0)
    if transformer_model_dim == 0 : # Use RNN output dim
        transformer_model_dim = x.shape[-1]
    else: # Project to specified dimension
        x = Dense(transformer_model_dim, activation=config['cnn_activation'], # Using cnn_activation for projection too
                  kernel_regularizer=l1_l2(l2=config['l2_reg']),
                  name=f"{biomarker_name}_hybrid_transformer_projection")(x)
    
    val_dim = int(config['transformer_key_dim'] * config.get('transformer_value_dim_mult', 1.0))
    ff_dim = int(transformer_model_dim * config.get('transformer_ff_dim_mult', 4.0)) # Common practice: ff_dim = 4 * model_dim

    for i in range(config['transformer_num_blocks']):
        x = AttentionBlock(num_heads=config['transformer_num_heads'], key_dim=config['transformer_key_dim'],
                           value_dim=val_dim, ff_dim=ff_dim, dropout_rate=config['transformer_dropout'],
                           l2_reg=config['l2_reg'], name=f"{biomarker_name}_hybrid_transformer_block_{i}")(x)
    
    # Pooling after Transformer
    if config['global_pooling_type_hybrid'] == 'avg': x = GlobalAveragePooling1D()(x)
    elif config['global_pooling_type_hybrid'] == 'max': x = GlobalMaxPooling1D()(x)
    elif config['global_pooling_type_hybrid'] == 'avgmax':
        x = Concatenate()([GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)])
    else: x = Flatten()(x)


    # Final Dense Layers
    for units in config['final_dense_units']:
        x = Dense(units, kernel_regularizer=l1_l2(l2=config['l2_reg']), name=f"{biomarker_name}_hybrid_dense_{units}")(x)
        x = BatchNormalization(name=f"{biomarker_name}_hybrid_bn_dense_{units}")(x)
        x = Activation(config['dense_activation'])(x)
        x = Dropout(config['dropout_final_dense'], name=f"{biomarker_name}_hybrid_drop_dense_{units}")(x)

    raw_output = Dense(num_outputs, activation='linear', name=f"{biomarker_name}_hybrid_raw_logits")(x)
    constrained_output = PhysiologicalConstraintLayer(biomarker_name=biomarker_name,
                                                     activation_type=config['output_activation_type'],
                                                     name=f"{biomarker_name}_hybrid_constrained_output")(raw_output)

    model = Model(inputs=inputs, outputs=constrained_output, name=f"{biomarker_name}_HybridCNNRNNTransformer")
    return model

# =============== PHYSICS-INFORMED DEEP LEARNING MODEL BUILDERS (Finalized) ===============

def build_physics_informed_model(input_shape_features: Tuple, num_outputs: int, biomarker_name: str,
                                 config: Dict) -> Model:
    """
    Generic builder for physics-informed models.
    It creates a feature-driven path and one or more physics-driven paths, then combines them.
    The exact structure of physics paths depends on 'biomarker_name' and 'config'.
    Config keys:
        - 'base_model_type': 'cnn', 'rnn', 'hybrid' (to build the feature processing backbone)
        - 'base_model_config': config dict for the chosen base_model_type builder
        - 'physics_paths': List of dicts, each describing a physics path.
            - path_config: {'type': 'bll'/'scattering', 'params': {... specific to BLL/ScatteringLayer ...},
                            'feature_map_dense_units': [..], 'feature_map_activation': 'relu'}
        - 'combination_method': 'average', 'weighted_sum_learnable', 'concatenate_dense'
        - 'final_dense_units_combiner': [..] (if combination_method is 'concatenate_dense')
        - 'l2_reg': float, 'dropout_combiner': float
        - 'output_activation_type': str
    """
    inputs = Input(shape=input_shape_features, name=f"{biomarker_name}_pi_input")
    
    # 1. Shared Feature Extractor Backbone or Direct Feature-Driven Path
    # This backbone can be used by all paths or just the direct data-driven one.
    # For simplicity, let's make a main data-driven path and then separate mappers for physics paths.
    
    # Direct Data-Driven Path
    if config['base_model_type'] == 'cnn':
        # We need a way to get the "logit" equivalent from the base model before final constraint
        temp_cnn_config = config['base_model_config'].copy()
        temp_cnn_config['output_activation_type'] = 'linear' # Get raw logit
        base_model_direct = build_spectral_attention_cnn(input_shape_features, num_outputs, f"{biomarker_name}_direct", temp_cnn_config)
        direct_logit = base_model_direct(inputs)
    elif config['base_model_type'] == 'rnn':
        temp_rnn_config = config['base_model_config'].copy()
        temp_rnn_config['output_activation_type'] = 'linear'
        base_model_direct = build_temporal_attention_lstm(input_shape_features, num_outputs, f"{biomarker_name}_direct", temp_rnn_config)
        direct_logit = base_model_direct(inputs)
    elif config['base_model_type'] == 'hybrid':
        temp_hybrid_config = config['base_model_config'].copy()
        temp_hybrid_config['output_activation_type'] = 'linear'
        base_model_direct = build_hybrid_cnn_lstm_transformer(input_shape_features, num_outputs, f"{biomarker_name}_direct", temp_hybrid_config)
        direct_logit = base_model_direct(inputs)
    else: # Simple MLP for direct path if no complex base model
        x_direct = inputs
        for units in config.get('direct_mlp_units', [64,32]):
            x_direct = Dense(units, activation='relu', kernel_regularizer=l1_l2(l2=config['l2_reg']))(x_direct)
        direct_logit = Dense(num_outputs, activation='linear', name=f"{biomarker_name}_direct_logit")(x_direct)

    all_path_outputs = [direct_logit] # Start with the direct data-driven path output (logit)

    # 2. Physics-Driven Paths
    for i, path_config in enumerate(config.get('physics_paths', [])):
        path_type = path_config['type'].lower()
        params = path_config['params']
        
        # MLP to map input features to inputs required by the physics layer
        x_physics_map = inputs
        for units in path_config.get('feature_map_dense_units', [32, 16]):
            x_physics_map = Dense(units, activation=path_config.get('feature_map_activation', 'relu'),
                                  kernel_regularizer=l1_l2(l2=config['l2_reg']))(x_physics_map)
        
        if path_type == 'bll': # Beer-Lambert Law Path
            # x_physics_map should now output estimated absorbances
            num_bll_wavelengths = len(params['wavelengths_nm_str'])
            estimated_absorbances = Dense(num_bll_wavelengths, activation='softplus', # Absorbance must be > 0
                                          name=f"{biomarker_name}_pi_est_abs_{i}")(x_physics_map)
            
            bll_layer = BeerLambertLawLayer(name=f"{biomarker_name}_pi_bll_{i}", **params)
            # concentrations_tensor shape (batch, num_chromophores_in_this_bll_config)
            concentrations_tensor = bll_layer(estimated_absorbances)
            
            # We need to map these chromophore concentrations to the target biomarker concentration
            # This mapping depends on what chromophores BLL is configured for and what the target biomarker is.
            # Example: If BLL outputs [HbO2, Hb] and target is SpO2:
            if biomarker_name.lower() == 'spo2' and 'hb_oxy' in bll_layer.chromophore_names and 'hb_deoxy' in bll_layer.chromophore_names:
                idx_hbo2 = bll_layer.chromophore_names.index('hb_oxy')
                idx_hb = bll_layer.chromophore_names.index('hb_deoxy')
                hbo2_c = concentrations_tensor[:, idx_hbo2:idx_hbo2+1]
                hb_c = concentrations_tensor[:, idx_hb:idx_hb+1]
                epsilon = 1e-7
                physics_ratio = tf.maximum(hbo2_c, 0.0) / (tf.maximum(hbo2_c + hb_c, epsilon))
                # Convert ratio to logit
                physics_logit = tf.math.log(physics_ratio / (1.0 - physics_ratio + epsilon) + epsilon)
                physics_logit = tf.where(tf.math.is_finite(physics_logit), physics_logit, tf.zeros_like(physics_logit))
                all_path_outputs.append(physics_logit)
            else: # General case: assume one of the BLL chromophores is the target or use a dense layer to map
                  # If BLL configured for 'glucose_mg_dl' and biomarker_name is 'glucose'
                if f"{biomarker_name.lower()}_mg_dl" in bll_layer.chromophore_names: # Assuming this convention
                    idx_target = bll_layer.chromophore_names.index(f"{biomarker_name.lower()}_mg_dl")
                    target_conc = concentrations_tensor[:, idx_target:idx_target+1]
                elif len(bll_layer.chromophore_names) == 1: # If only one chromophore, assume it's the target
                    target_conc = concentrations_tensor
                else: # Multiple chromophores, need to map to single biomarker output with a Dense layer
                    target_conc = Dense(num_outputs, activation='linear', name=f"{biomarker_name}_pi_bll_map_{i}")(concentrations_tensor)
                
                # Convert target_conc (which is a concentration) to a logit for combination
                # This is tricky. For now, just pass it through another dense layer.
                physics_output_logit = Dense(num_outputs, activation='linear', name=f"{biomarker_name}_pi_bll_logit_{i}")(target_conc)
                all_path_outputs.append(physics_output_logit)

        elif path_type == 'scattering': # Scattering Model Path
            scattering_layer = ScatteringModelLayer(name=f"{biomarker_name}_pi_scatter_{i}", **params)
            # concentrations_tensor shape (batch, num_scattering_biomarkers_in_this_config)
            concentrations_tensor = scattering_layer(x_physics_map) # x_physics_map are features for scattering layer
            
            # Similar mapping from scattering_layer's biomarkers to the target biomarker
            if biomarker_name.lower() in scattering_layer.all_scattering_biomarkers: # If target directly output
                idx_target = scattering_layer.all_scattering_biomarkers.index(biomarker_name.lower())
                target_conc = concentrations_tensor[:, idx_target:idx_target+1]
            elif len(scattering_layer.all_scattering_biomarkers) == 1:
                target_conc = concentrations_tensor
            else:
                target_conc = Dense(num_outputs, activation='linear', name=f"{biomarker_name}_pi_scatter_map_{i}")(concentrations_tensor)
            
            physics_output_logit = Dense(num_outputs, activation='linear', name=f"{biomarker_name}_pi_scatter_logit_{i}")(target_conc)
            all_path_outputs.append(physics_output_logit)
        # Add other physics path types here...

    # 3. Combine outputs from all paths
    if len(all_path_outputs) == 1:
        combined_logit = all_path_outputs[0]
    else:
        if config['combination_method'] == 'average':
            combined_logit = Average(name=f"{biomarker_name}_pi_average_logits")(all_path_outputs)
        elif config['combination_method'] == 'weighted_sum_learnable':
            # Stack outputs and use a Dense layer with 1 output and no bias to learn weights
            stacked_outputs = Concatenate(axis=-1, name=f"{biomarker_name}_pi_stack_logits")(all_path_outputs) # (batch, num_paths * num_outputs)
            if num_outputs > 1: # Need to handle multi-output averaging carefully
                # Reshape to (batch, num_paths, num_outputs) then learn weights per output
                reshaped_for_weighting = Reshape((len(all_path_outputs), num_outputs))(stacked_outputs)
                # This needs custom weighting. Simpler: learn dense combination.
                combined_logit = Dense(num_outputs, activation='linear', name=f"{biomarker_name}_pi_dense_combiner")(stacked_outputs)

            else: # Single output case
                # Weights should sum to 1, or just learn a linear combination
                # Using a simple Dense layer to learn the combination
                combined_logit = Dense(num_outputs, activation='linear', use_bias=False, # No bias for weighted sum
                                   kernel_initializer=tf.keras.initializers.Constant(1.0/len(all_path_outputs)),
                                   name=f"{biomarker_name}_pi_weighted_sum_layer")(stacked_outputs)

        elif config['combination_method'] == 'concatenate_dense':
            concatenated_outputs = Concatenate(name=f"{biomarker_name}_pi_concat_logits")(all_path_outputs)
            x_comb = concatenated_outputs
            for units in config.get('final_dense_units_combiner', [32]):
                x_comb = Dense(units, activation='relu', kernel_regularizer=l1_l2(l2=config['l2_reg']))(x_comb)
                x_comb = Dropout(config.get('dropout_combiner', 0.1))(x_comb)
            combined_logit = Dense(num_outputs, activation='linear', name=f"{biomarker_name}_pi_final_combined_logit")(x_comb)
        else:
            raise ValueError(f"Unknown combination_method: {config['combination_method']}")

    # 4. Final Output Constraint
    final_output = PhysiologicalConstraintLayer(biomarker_name=biomarker_name,
                                               activation_type=config['output_activation_type'],
                                               name=f"{biomarker_name}_pi_constrained_output")(combined_logit)

    model = Model(inputs=inputs, outputs=final_output, name=f"{biomarker_name}_PhysicsInformedGenericModel")
    return model


class PhysicsInformedBiomarkerPredictor:
    """
    Main class for the Physics-Informed Optical Biomarker Prediction Framework.
    Orchestrates preprocessing, model training, prediction, and evaluation.
    """
    def __init__(self, biomarker_type: str, config: Optional[Dict] = None):
        self.biomarker_type = biomarker_type.lower().replace(" ", "_")
        self.user_config = config if config else {}
        self.config = self._merge_configs(self._get_default_config(), self.user_config)
        
        self.preprocessor_pipeline_: Optional[Pipeline] = None
        self.models_: Dict[str, Union[Model, BaseEstimator]] = {} # Holds trained models
        self.model_weights_: Optional[Dict[str, float]] = None # For weighted ensembling
        self.best_model_key_: Optional[str] = None
        self.is_fitted_: bool = False
        self.training_history_: Dict[str, Any] = {} # Store training history, metrics
        self.feature_names_in_: Optional[List[str]] = None # Original feature names before preprocessing
        self.feature_names_processed_: Optional[List[str]] = None # Feature names after preprocessing pipeline
        self.feature_importances_: Optional[pd.DataFrame] = None # Aggregated feature importances

        self._setup_environment()

    def _setup_environment(self):
        """Sets up random seeds and GPU configuration."""
        seed = self.config['general']['random_state']
        np.random.seed(seed)
        tf.random.set_seed(seed)
        # Python's random module if used elsewhere:
        # import random
        # random.seed(seed)

        if self.config['general']['gpu_memory_growth']:
            try:
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info(f"GPU memory growth enabled for {len(gpus)} GPUs.")
            except Exception as e:
                logging.warning(f"Could not configure GPU memory growth: {e}")

    def _get_default_config(self) -> Dict:
        """Provides a comprehensive default configuration."""
        return {
            "general": {
                "random_state": 42,
                "verbose": 1, # 0: silent, 1: info, 2: debug
                "n_jobs": -1, # For scikit-learn parallel tasks
                "gpu_memory_growth": True,
                "results_dir": "biomarker_predictor_results"
            },
            "data_preprocessing": {
                "preprocessor_active": True,
                "steps": [ # List of preprocessor steps with their configs
                    {"name": "signal_preprocessor", "active": True, "params": {
                        "signal_column_prefixes": ['red', 'ir', 'green', 'blue', 'raw_signal'], # Match actual raw signal column names/prefixes
                        "sampling_rate_hz": 100.0,
                        "outlier_capping_method": 'iqr', "outlier_threshold": 3.0,
                        "filter_type": 'bandpass', "lowcut_hz": 0.5, "highcut_hz": 8.0, "filter_order": 4,
                        "detrend_method": 'polynomial', "poly_detrend_degree": 2,
                        "normalization_method": 'zscore', # Applied per segment/signal
                        "segment_length_s": 10.0, "segment_overlap_ratio": 0.5,
                        "output_prefix": 'proc_', "keep_original_columns": True
                    }},
                    {"name": "wavelet_features", "active": True, "params": {
                        "signal_column_glob": "proc_*", "wavelet_name": 'db8', "max_level": 5,
                        "features_to_extract": ['energy', 'entropy', 'std', 'mean', 'kurtosis', 'skew', 'approx_coeff_stats', 'detail_coeff_ratios'],
                        "output_prefix_per_signal_col": True
                    }},
                    {"name": "phase_space_features", "active": False, "params": { # Computationally expensive
                        "signal_column_glob": "proc_*", "embedding_delay_method": 'ami', "max_embedding_delay": 15,
                        "embedding_dimension_method": 'fnn', "max_embedding_dimension": 8,
                        "nolds_features": ['sampen', 'corr_dim', 'lyap_r', 'hurst_rs', 'dfa'],
                        "output_prefix_per_signal_col": True
                    }},
                    {"name": "spectral_features", "active": True, "params": {
                        "signal_column_glob": "proc_*", "sampling_rate_hz": 100.0, # Should match signal_preprocessor
                        "freq_bands_hz": {'vlf': (0.003, 0.04), 'lf': (0.04, 0.15), 'hf': (0.15, 0.4)},
                        "spectral_features": ['band_power_rel', 'band_peak_freq', 'spectral_entropy', 'spectral_centroid'],
                        "output_prefix_per_signal_col": True
                    }},
                    {"name": "hrv_features", "active": False, "params": { # Requires good peak detection
                        "signal_column_glob": "proc_ir", # Specify which processed signal to use for peaks
                        "sampling_rate_hz": 100.0, "peak_detection_method": 'simple_scipy',
                        "time_domain_features": ['mean_rr', 'sdnn', 'rmssd', 'pnn50'],
                        "freq_domain_features": ['lf_power', 'hf_power', 'lf_hf_ratio'],
                        "min_rri_count_for_hrv": 20,
                        "output_prefix_per_signal_col": False # HRV features are usually global
                    }},
                    {"name": "multispectral_features", "active": True, "params": {
                        # Assumes previous steps created features like 'proc_red_mean', 'proc_ir_ac_amp' etc.
                        # These prefixes should match outputs of previous feature extractors or be manually created input columns.
                        "wavelength_channel_prefixes": ['proc_red', 'proc_ir', 'proc_green'], # From signal_preprocessor
                        "mean_suffix": '_wavelet_app_L5_mean', # Example: if wavelet features are used
                        "std_suffix": '_wavelet_app_L5_std',   # And we want ratios of these
                        "ac_amplitude_suffix": '_ac_amplitude', # These would need to be explicitly calculated
                        "dc_offset_suffix": '_dc_offset',       # or be part of input features.
                        "biomarkers_of_interest": [self.biomarker_type], # Automatically set from class instance
                        "output_prefix": "msf_"
                    }},
                    {"name": "imputer", "active": True, "params": {"strategy": "median"}}, # SimpleImputer
                    {"name": "scaler", "active": True, "params": {"type": "robust"}} # robust, standard, minmax
                ]
            },
            "model_selection": {
                "architecture_type": "ensemble", # 'single_dl', 'single_ml', 'ensemble'
                # For 'single_dl' or 'single_ml', specify model_name from 'dl_models' or 'ml_models'
                "primary_model_name": "physics_informed_cnn",
                "dl_models": { # Configs for different DL architectures
                    "spectral_cnn": {"active": True, "builder": "build_spectral_attention_cnn", "config": {
                        "num_filters_list": [64, 128], "kernel_sizes_list": [7, 5], "res_strides_list": [1,1],
                        "res_activation": 'relu', "res_block_dropout": 0.1, "use_squeeze_excitation": False,
                        "attention_heads": 0, "attention_key_dim": 32, "global_pooling_type": 'avg',
                        "final_dense_units": [64], "dropout_final_dense": 0.2, "dense_activation": 'relu',
                        "l2_reg_cnn": 1e-5, "l2_reg_dense": 1e-4, "output_activation_type": 'scaled_sigmoid'
                    }},
                    "temporal_rnn": {"active": True, "builder": "build_temporal_attention_lstm", "config": {
                        "rnn_type": 'lstm', "rnn_units_list": [64, 32], "bidirectional": True,
                        "rnn_activation": 'tanh', "rnn_dropout": 0.2, "recurrent_dropout": 0.2,
                        "attention_after_rnn": True, "attention_heads": 2, "attention_key_dim": 32,
                        "global_pooling_type_rnn": 'avg',
                        "final_dense_units": [32], "dropout_final_dense": 0.2, "dense_activation": 'relu',
                        "l2_reg_rnn": 1e-5, "l2_reg_dense": 1e-4, "output_activation_type": 'scaled_sigmoid'
                    }},
                    "hybrid_model": {"active": False, "builder": "build_hybrid_cnn_lstm_transformer", "config": {
                        # ... (extensive config for hybrid model) ...
                        "cnn_num_filters_list": [32], "cnn_kernel_sizes_list": [5], "cnn_activation": 'relu', "cnn_dropout":0.1, "use_cnn_se":False,
                        "rnn_type":'lstm', "rnn_units_list": [32], "rnn_bidirectional":True, "rnn_activation":'tanh', "rnn_dropout":0.1, "rec_dropout_rnn":0.1,
                        "transformer_num_blocks":1, "transformer_num_heads":2, "transformer_key_dim":16, "transformer_dropout":0.1,
                        "global_pooling_type_hybrid": 'avg', "final_dense_units": [32], "dropout_final_dense":0.1, "dense_activation":'relu',
                        "l2_reg":1e-5, "output_activation_type": 'scaled_sigmoid'
                    }},
                    "physics_informed_cnn": {"active": True, "builder": "build_physics_informed_model", "config": {
                        "base_model_type": "cnn", # Backbone for feature processing
                        "base_model_config": { # Config for build_spectral_attention_cnn as backbone
                            "num_filters_list": [32, 64], "kernel_sizes_list": [5, 3], "res_strides_list": [1,1],
                            "res_activation": 'relu', "res_block_dropout": 0.1, "use_squeeze_excitation": False,
                            "attention_heads": 0, "global_pooling_type": 'avg',
                            "final_dense_units": [32], "dropout_final_dense": 0.1, "dense_activation": 'relu',
                            "l2_reg_cnn": 1e-5, "l2_reg_dense": 1e-4 # Output activation will be linear for PI model internal paths
                        },
                        "physics_paths": [ # Define physics paths based on biomarker
                            # Example for SpO2:
                            # {"type": "bll", "params": {"chromophore_configs": {...}, "wavelengths_nm_str": [...]}, 
                            #  "feature_map_dense_units": [16]}
                            # Example for Glucose:
                            # {"type": "scattering", "params": {"biomarker_ri_effects": {...}}, "feature_map_dense_units": [16]}
                        ],
                        "combination_method": 'concatenate_dense', "final_dense_units_combiner": [32, 16],
                        "l2_reg": 1e-5, "dropout_combiner": 0.1, "output_activation_type": 'scaled_sigmoid'
                    }}
                },
                "ml_models": { # Classical ML models
                    "random_forest": {"active": True, "model_class": RandomForestRegressor, "params": {
                        "n_estimators": 200, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 3, "random_state": "{{general.random_state}}"
                    }},
                    "gradient_boosting": {"active": True, "model_class": GradientBoostingRegressor, "params": {
                        "n_estimators": 200, "learning_rate": 0.05, "max_depth": 7, "subsample":0.8, "random_state": "{{general.random_state}}"
                    }},
                    "huber_regressor": {"active": False, "model_class": HuberRegressor, "params": {"epsilon": 1.35, "alpha":0.001}}
                    # Add SVR, KNN, etc.
                },
                "ensemble_strategy": { # For architecture_type 'ensemble'
                    "method": "stacking", # 'voting_regressor', 'stacking', 'weighted_average'
                    "voting_weights": None, # List of weights if 'voting_regressor' and weights are manual
                    "stacking_final_estimator": "ridge", # 'ridge', 'gbr', 'rf', or a full model_class key
                    "stacking_cv": 5,
                    "stacking_passthrough": True, # Use original features in meta-regressor
                    # For 'weighted_average', weights can be derived from validation performance.
                }
            },
            "training": {
                "optimizer": {"type": "adamw", "learning_rate": 1e-3, "weight_decay": 1e-4, "clipnorm": 1.0},
                "loss_function": "huber", # 'mse', 'mae', 'huber', or custom
                "metrics": ['mae', 'mse'], # Additional metrics for Keras compile
                "batch_size": 64,
                "epochs": 200,
                "validation_split": 0.2, # Used if no explicit validation set is provided to fit()
                "callbacks": {
                    "early_stopping": {"active": True, "monitor": "val_loss", "patience": 20, "restore_best_weights": True},
                    "reduce_lr_on_plateau": {"active": True, "monitor": "val_loss", "factor": 0.2, "patience": 10, "min_lr": 1e-6},
                    "model_checkpoint": {"active": True, "monitor": "val_loss", "save_best_only": True, "filename_template": "{model_name}_best.h5"}
                },
                "data_augmentation": {"active": False, "methods": []}, # e.g., add_noise, time_shift
                "use_kfold_cv": False, # If True, fit performs k-fold CV; if False, simple train/val split
                "n_splits_kfold": 5,
                "kfold_shuffle": True,
                "kfold_group_by_col": None # Column name for GroupKFold
            },
            "hyperparameter_optimization": {
                "active": False,
                "ray_tune_config": {
                    "scheduler": "asha", # 'asha', 'pbt', 'hyperband'
                    "search_alg": "hyperopt", # 'hyperopt', 'optuna', 'bohb', 'random'
                    "num_samples": 50, # Number of HPO trials
                    "metric_to_optimize": "val_mae", "optimization_mode": "min",
                    "resources_per_trial": {"cpu": 2, "gpu": 0.5}, # Adjust based on hardware
                    "max_concurrent_trials": 0, # 0 means default (num_cpus / cpus_per_trial)
                    # Parameter search space defined per model in its own section or dynamically.
                    "param_space_common_dl": { # Common search space for DL models
                        "learning_rate": {"type": "loguniform", "lower": 1e-5, "upper": 1e-2},
                        "batch_size": {"type": "choice", "values": [32, 64, 128]},
                        "dropout_rate_dense": {"type": "uniform", "lower": 0.1, "upper": 0.5},
                    },
                    # Specific HPO spaces can be added for each model type.
                }
            },
            "evaluation": {
                "primary_metric": "mae", # 'mae', 'rmse', 'r2'
                "clarke_error_grid_for_glucose": True,
                "spo2_accuracy_thresholds_percent": [2.0, 4.0],
                "bland_altman_plot": True
            },
            "uncertainty_estimation": { # For DL models predominantly
                "method": "mc_dropout", # 'mc_dropout', 'ensemble_variance', 'none'
                "mc_dropout_samples": 30 # Number of forward passes for MC Dropout
            },
            "saving_loading":{
                "compress_joblib": 3 # Compression level for joblib.dump
            }
        }

    def _merge_configs(self, base_cfg: Dict, user_cfg: Dict) -> Dict:
        """Recursively merges user config into base config."""
        merged = base_cfg.copy()
        for key, value in user_cfg.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self._merge_configs(merged[key], value)
            elif isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                # Handle simple template {{section.key}}
                path = value[2:-2].split('.')
                resolved_value = _get_from_nested_dict(merged, path) # Resolve from already merged part or base
                if resolved_value is None: resolved_value = _get_from_nested_dict(user_cfg, path) # Try user_cfg again
                if resolved_value is not None:
                    merged[key] = resolved_value
                else:
                    warn(f"Config template {value} could not be resolved. Keeping as string.")
                    merged[key] = value # Keep as string if not found
            else:
                merged[key] = value
        return merged

    def _initialize_preprocessor_pipeline(self) -> Optional[Pipeline]:
        """Builds the scikit-learn preprocessing pipeline based on config."""
        if not self.config['data_preprocessing']['preprocessor_active']:
            return None

        steps = []
        for step_config in self.config['data_preprocessing']['steps']:
            if not step_config.get('active', False):
                continue
            
            name = step_config['name']
            params = step_config.get('params', {})

            # Resolve any template variables in params
            for p_key, p_val in params.items():
                if isinstance(p_val, str) and p_val.startswith("{{") and p_val.endswith("}}"):
                    path = p_val[2:-2].split('.')
                    # Resolve from main config. Pass self.config to resolve from anywhere.
                    resolved_val = _get_from_nested_dict(self.config, path)
                    if resolved_val is not None: params[p_key] = resolved_val
                    else: warn(f"Template {p_val} in preprocessor params for {name} not resolved.")
            
            # Link sampling_rate from signal_preprocessor to other extractors if needed
            if name != "signal_preprocessor" and "sampling_rate_hz" in params:
                sp_conf = next((s for s in self.config['data_preprocessing']['steps'] if s['name'] == 'signal_preprocessor'), None)
                if sp_conf and 'sampling_rate_hz' in sp_conf['params']:
                    params['sampling_rate_hz'] = sp_conf['params']['sampling_rate_hz']


            transformer = None
            if name == "signal_preprocessor": transformer = PPGSignalPreprocessor(**params)
            elif name == "wavelet_features":
                if has_pywavelets: transformer = WaveletFeatureExtractor(**params)
                else: logging.warning("Wavelet features skipped: PyWavelets not installed."); continue
            elif name == "phase_space_features":
                if has_nolds: transformer = PhaseSpaceTransformer(**params)
                else: logging.warning("Phase space features skipped: Nolds not installed."); continue
            elif name == "spectral_features": transformer = SpectralFeatureExtractor(**params)
            elif name == "hrv_features": transformer = HeartRateVariabilityTransformer(**params)
            elif name == "multispectral_features":
                # Ensure biomarker_type is correctly passed if it's a param
                if 'biomarkers_of_interest' in params and isinstance(params['biomarkers_of_interest'], list):
                    # If the config has a placeholder like [self.biomarker_type], it needs to be resolved.
                    # This is tricky with _get_default_config. Simpler to handle it here or ensure it's already resolved.
                    # Assuming it's correctly set if not a template.
                    pass
                transformer = MultiSpectralFeatureTransformer(**params)
            elif name == "imputer": transformer = SimpleImputer(**params)
            elif name == "scaler":
                scaler_type = params.pop("type", "robust").lower()
                if scaler_type == "robust": transformer = RobustScaler(**params)
                elif scaler_type == "standard": transformer = StandardScaler(**params)
                elif scaler_type == "minmax": transformer = MinMaxScaler(**params)
                else: logging.warning(f"Unknown scaler type: {scaler_type}. Skipping scaler."); continue
            # Add more custom or sklearn transformers here
            else:
                logging.warning(f"Unknown preprocessor step name: {name}. Skipping.")
                continue
            
            if transformer:
                steps.append((name, transformer))
        
        if not steps: return None
        return Pipeline(steps)

    def _resolve_config_placeholders(self, target_dict: Dict) -> Dict:
        """Recursively resolve {{...}} placeholders within a dictionary."""
        resolved_dict = {}
        for key, value in target_dict.items():
            if isinstance(value, dict):
                resolved_dict[key] = self._resolve_config_placeholders(value)
            elif isinstance(value, list):
                resolved_dict[key] = [self._resolve_config_placeholders(item) if isinstance(item, dict) else item for item in value]
            elif isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                path = value[2:-2].split('.')
                resolved_value = _get_from_nested_dict(self.config, path)
                if resolved_value is not None:
                    resolved_dict[key] = resolved_value
                else:
                    warn(f"Config placeholder {value} in target_dict could not be resolved. Keeping as string.")
                    resolved_dict[key] = value
            else:
                resolved_dict[key] = value
        return resolved_dict

    def _build_dl_model(self, model_name: str, dl_model_config: Dict, input_shape: Tuple, num_outputs: int) -> Model:
        """Builds a specific DL model based on its configuration."""
        builder_func_name = dl_model_config['builder']
        specific_config = self._resolve_config_placeholders(dl_model_config['config'])

        # Dynamically get builder function from globals (this script's context)
        builder_func = globals().get(builder_func_name)
        if builder_func is None or not callable(builder_func):
            raise ValueError(f"DL model builder function '{builder_func_name}' not found or not callable.")

        # Special handling for physics_informed_model builder as it needs more dynamic param setup
        if builder_func_name == "build_physics_informed_model":
            # Dynamically populate physics_paths for the current biomarker
            physics_paths_cfg = []
            if self.biomarker_type == 'spo2':
                # Example BLL path for SpO2
                hb_oxy_coeffs = _get_from_nested_dict(OPTICAL_CONSTANTS, ['hemoglobin', 'oxy_extinction_coefficients'], {})
                hb_deoxy_coeffs = _get_from_nested_dict(OPTICAL_CONSTANTS, ['hemoglobin', 'deoxy_extinction_coefficients'], {})
                bll_wl_spo2 = specific_config.get('bll_wavelengths_spo2', ['660nm', '940nm']) # Default or from config
                
                path_params_spo2 = {
                    "chromophore_configs": {
                        'hb_oxy': {wl: hb_oxy_coeffs.get(wl,0) for wl in bll_wl_spo2},
                        'hb_deoxy': {wl: hb_deoxy_coeffs.get(wl,0) for wl in bll_wl_spo2}
                    },
                    "wavelengths_nm_str": bll_wl_spo2,
                    "initial_base_path_length_mm": specific_config.get('bll_path_length_spo2_mm', 5.0),
                    "learnable_dpf": specific_config.get('bll_learnable_dpf_spo2', True)
                }
                physics_paths_cfg.append({
                    "type": "bll", "params": path_params_spo2,
                    "feature_map_dense_units": specific_config.get('bll_feature_map_units_spo2', [16])
                })
            elif self.biomarker_type == 'glucose':
                # Example Scattering path for Glucose
                glucose_ri_effect = {'glucose_mg_dl': OPTICAL_CONSTANTS['glucose']['refractive_index_change_per_mg_dl']}
                path_params_glucose_scatter = {
                    "biomarker_ri_effects": glucose_ri_effect,
                    "base_medium_ri": OPTICAL_CONSTANTS['tissue_properties']['baseline_refractive_index_medium'],
                    "learnable_sensitivities": specific_config.get('scatter_learnable_sens_glucose', True)
                }
                physics_paths_cfg.append({
                    "type": "scattering", "params": path_params_glucose_scatter,
                    "feature_map_dense_units": specific_config.get('scatter_feature_map_units_glucose', [32,16])
                })
                # Optionally add BLL path for glucose if configured
                if specific_config.get('use_glucose_absorption_path', False):
                    glucose_abs_coeffs = _get_from_nested_dict(OPTICAL_CONSTANTS, ['glucose', 'specific_absorption_cm_inv_per_mg_dl'], {})
                    bll_wl_glucose = specific_config.get('bll_wavelengths_glucose_abs', ['1550nm', '2100nm'])
                    path_params_glucose_abs = {
                         "chromophore_configs": {'glucose_mg_dl': {wl: glucose_abs_coeffs.get(wl,0) for wl in bll_wl_glucose}},
                         "wavelengths_nm_str": bll_wl_glucose,
                         "initial_base_path_length_mm": specific_config.get('bll_path_length_glucose_mm', 2.0)
                    }
                    physics_paths_cfg.append({
                        "type":"bll", "params": path_params_glucose_abs,
                        "feature_map_dense_units": specific_config.get('bll_feature_map_units_glucose_abs', [16])
                    })
            # Add similar logic for other biomarkers (hemoglobin, lipids/cholesterol)
            
            specific_config['physics_paths'] = physics_paths_cfg # Update the config for the builder
            model = builder_func(input_shape, num_outputs, self.biomarker_type, specific_config)
        else: # For other standard builders
            model = builder_func(input_shape, num_outputs, self.biomarker_type, specific_config)
        
        return model

    def _build_ml_model(self, model_name: str, ml_model_config: Dict) -> BaseEstimator:
        """Builds a specific classical ML model based on its configuration."""
        ModelClass = ml_model_config['model_class'] # This should be the actual class object
        params = self._resolve_config_placeholders(ml_model_config.get('params', {}))
        
        # For ensemble models, they might need base estimators
        if ModelClass == StackingRegressor:
             # Stacking requires 'estimators' list of (name, estimator) tuples
             # And 'final_estimator'. These need to be resolved from config too.
            estimators_config = params.pop('estimators_config', []) # e.g. [{'name':'rf', 'key':'random_forest'}, ...]
            base_estimators = []
            for est_conf in estimators_config:
                base_model_key = est_conf['key'] # Key in self.config['model_selection']['ml_models']
                base_model_full_config = self.config['model_selection']['ml_models'].get(base_model_key)
                if base_model_full_config and base_model_full_config['active']:
                     base_est_class = base_model_full_config['model_class']
                     base_est_params = self._resolve_config_placeholders(base_model_full_config.get('params', {}))
                     base_estimators.append((est_conf['name'], base_est_class(**base_est_params)))
                else: warn(f"Base estimator {base_model_key} for StackingRegressor not found or inactive.")
            
            final_est_key = params.pop('final_estimator_key', 'ridge') # Default final estimator
            final_est_config = self.config['model_selection']['ml_models'].get(final_est_key)
            if final_est_config and final_est_config['active']:
                final_est_class = final_est_config['model_class']
                final_est_params = self._resolve_config_placeholders(final_est_config.get('params', {}))
                final_estimator_instance = final_est_class(**final_est_params)
            else: # Fallback for final estimator
                warn(f"Final estimator {final_est_key} for StackingRegressor not found. Using Ridge default.")
                final_estimator_instance = Ridge()
            
            return StackingRegressor(estimators=base_estimators, final_estimator=final_estimator_instance, **params)
        
        elif ModelClass == VotingRegressor:
            estimators_config = params.pop('estimators_config', [])
            named_estimators = []
            for est_conf in estimators_config:
                base_model_key = est_conf['key']
                base_model_full_config = self.config['model_selection']['ml_models'].get(base_model_key)
                if base_model_full_config and base_model_full_config['active']:
                     base_est_class = base_model_full_config['model_class']
                     base_est_params = self._resolve_config_placeholders(base_model_full_config.get('params', {}))
                     named_estimators.append((est_conf['name'], base_est_class(**base_est_params)))
            return VotingRegressor(estimators=named_estimators, **params) # Pass other params like weights

        # Standard model instantiation
        return ModelClass(**params)
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
            X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_val: Optional[Union[pd.Series, np.ndarray]] = None,
            groups: Optional[Union[pd.Series, np.ndarray]] = None, # For GroupKFold
            run_hpo: Optional[bool] = None # Override HPO config for this fit call
           ) -> 'PhysicsInformedBiomarkerPredictor':
        """
        Fits the biomarker predictor to the training data.
        Handles preprocessing, hyperparameter optimization (optional), model training,
        and ensemble building.
        """
        if self.config['general']['verbose'] > 0:
            logging.info(f"Starting `fit` process for biomarker: {self.biomarker_type}")

        self.is_fitted_ = False # Reset fit status
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name=self.biomarker_type)
        
        self.feature_names_in_ = list(X.columns)
        
        # Ensure y is a Series for consistent handling
        if not isinstance(y, pd.Series): y = pd.Series(y)

        # 1. Hyperparameter Optimization (Optional)
        hpo_active = run_hpo if run_hpo is not None else self.config['hyperparameter_optimization']['active']
        if hpo_active and has_ray_tune:
            if self.config['general']['verbose'] > 0: logging.info("Running Hyperparameter Optimization with Ray Tune...")
            best_hpo_config = self._run_hpo_ray_tune(X, y, X_val, y_val, groups)
            if best_hpo_config:
                # Update self.config with the best hyperparameters found
                # This needs careful merging, e.g., update model-specific configs
                logging.info(f"HPO found best config: {best_hpo_config}")
                # Example: self.config['model_selection']['dl_models']['spectral_cnn']['config'].update(best_hpo_config['spectral_cnn'])
                # This requires a structured way to map HPO results back. For now, assume HPO modifies a copy or
                # the user manually updates based on HPO logs. A more integrated approach would update self.config.
                # For this implementation, let's assume HPO run separately produces a config to use.
                # Or, HPO directly optimizes parameters of the model(s) to be trained next.
                # For now, we'll train with the current self.config, which might have been updated by a prior HPO run.
                pass # Placeholder for sophisticated config update logic post-HPO
            else:
                logging.warning("HPO did not return a best configuration. Proceeding with current config.")
        elif hpo_active and not has_ray_tune:
            logging.warning("Hyperparameter optimization is active in config, but Ray Tune is not installed. Skipping HPO.")

        # 2. Initialize Preprocessor Pipeline
        self.preprocessor_pipeline_ = self._initialize_preprocessor_pipeline()

        # 3. Preprocess Data
        if self.preprocessor_pipeline_:
            if self.config['general']['verbose'] > 0: logging.info("Applying preprocessing pipeline to training data...")
            X_processed = self.preprocessor_pipeline_.fit_transform(X, y)
            # Get feature names after preprocessing
            try:
                self.feature_names_processed_ = self.preprocessor_pipeline_.get_feature_names_out()
            except Exception: # Some custom transformers might not have perfect get_feature_names_out
                 if hasattr(X_processed, 'columns'): self.feature_names_processed_ = list(X_processed.columns)
                 else: self.feature_names_processed_ = [f"proc_feat_{i}" for i in range(X_processed.shape[1])]

            if X_val is not None:
                if isinstance(X_val, np.ndarray): X_val = pd.DataFrame(X_val, columns=self.feature_names_in_)
                X_val_processed = self.preprocessor_pipeline_.transform(X_val)
            else: # If no X_val, need to split X_processed for internal validation during training
                X_val_processed = None
        else: # No preprocessing
            X_processed = X.copy()
            self.feature_names_processed_ = self.feature_names_in_
            if X_val is not None:
                 X_val_processed = X_val.copy() if isinstance(X_val, pd.DataFrame) else pd.DataFrame(X_val, columns=self.feature_names_in_)
            else:
                 X_val_processed = None
        
        # Convert to numpy for Keras/sklearn models if it's a DataFrame
        if isinstance(X_processed, pd.DataFrame): X_processed_np = X_processed.to_numpy()
        else: X_processed_np = X_processed # Assume already numpy or compatible
        
        if y_val is not None and isinstance(y_val, np.ndarray): y_val = pd.Series(y_val)

        if X_val_processed is not None:
            if isinstance(X_val_processed, pd.DataFrame): X_val_processed_np = X_val_processed.to_numpy()
            else: X_val_processed_np = X_val_processed

        # 4. Model Training
        # Determine if using K-Fold CV or a single train/validation split
        use_kfold = self.config['training']['use_kfold_cv']
        
        # Store validation scores for model selection / ensemble weighting
        self.model_validation_scores_ = {}

        if use_kfold:
            # K-Fold CV requires training models multiple times.
            # This implementation will focus on training a single instance of each model for simplicity here.
            # A full K-Fold CV loop would wrap the model building and training for each fold.
            # For now, if K-Fold is true, we'll just use the first fold from a split for demonstration.
            logging.warning("K-Fold CV is enabled, but this simplified `fit` will train on one primary split. Full CV needs a more complex loop.")
            # Fallback to train/val split logic if X_val/y_val not provided by KFold logic here
            if X_val is None and y_val is None: # Need to create a validation set from X_processed
                X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
                    X_processed_np, y.to_numpy(),
                    test_size=self.config['training']['validation_split'],
                    random_state=self.config['general']['random_state'],
                    shuffle=self.config['training']['kfold_shuffle']
                )
                y_val_fold_series = pd.Series(y_val_fold) # For evaluation metrics
            elif X_val_processed is not None and y_val is not None: # User provided X_val, y_val
                X_train_fold, y_train_fold = X_processed_np, y.to_numpy()
                X_val_fold, y_val_fold_series = X_val_processed_np, y_val
            else: # Should not happen
                raise ValueError("Invalid data state for training.")

        else: # Single train/validation split
            if X_val_processed is None or y_val is None:
                X_train_fold, X_val_fold, y_train_fold, y_val_fold_temp = train_test_split(
                    X_processed_np, y.to_numpy(),
                    test_size=self.config['training']['validation_split'],
                    random_state=self.config['general']['random_state'],
                    shuffle=True # Always shuffle for train/test split
                )
                y_val_fold_series = pd.Series(y_val_fold_temp)
            else:
                X_train_fold, y_train_fold = X_processed_np, y.to_numpy()
                X_val_fold, y_val_fold_series = X_val_processed_np, y_val.to_numpy() if isinstance(y_val, pd.Series) else y_val
                y_val_fold_series = pd.Series(y_val_fold_series)


        # Get input shape for DL models from the processed training data
        # Handle cases where X_train_fold might be 1D if only one sample after bad splitting
        if X_train_fold.ndim == 1:
            # This can happen if train_test_split results in a single sample for X_train_fold
            # Example: X_processed_np has shape (N, features), y has shape (N,).
            # If N is small, test_size could lead to X_train_fold being (features,)
            # We need to reshape it to (1, features)
            if X_processed_np.ndim > 1 and X_train_fold.shape[0] == X_processed_np.shape[1]:
                 X_train_fold = X_train_fold.reshape(1, -1)
            else: # Unclear shape, this is an issue
                 raise ValueError(f"X_train_fold has unexpected shape {X_train_fold.shape} from input {X_processed_np.shape}")

        dl_input_shape = X_train_fold.shape[1:] # (num_features_processed,) or (seq_len, num_features_processed)
        num_outputs = 1 # Assuming single biomarker regression output

        # Train selected DL models
        for model_key, dl_config in self.config['model_selection']['dl_models'].items():
            if dl_config.get('active', False):
                if self.config['general']['verbose'] > 0: logging.info(f"Training DL model: {model_key}")
                tf.keras.backend.clear_session() # Clear previous model graphs for memory
                model_instance = self._build_dl_model(model_key, dl_config, dl_input_shape, num_outputs)
                
                history, val_metrics = self._train_keras_model(
                    model_instance, model_key,
                    X_train_fold, y_train_fold,
                    X_val_fold, y_val_fold_series.to_numpy() if y_val_fold_series is not None else None
                )
                self.models_[model_key] = model_instance
                self.training_history_[model_key] = history
                if val_metrics: self.model_validation_scores_[model_key] = val_metrics

        # Train selected ML models
        for model_key, ml_config in self.config['model_selection']['ml_models'].items():
            if ml_config.get('active', False):
                if self.config['general']['verbose'] > 0: logging.info(f"Training ML model: {model_key}")
                model_instance = self._build_ml_model(model_key, ml_config)
                try:
                    model_instance.fit(X_train_fold, y_train_fold)
                    self.models_[model_key] = model_instance
                    if X_val_fold is not None and y_val_fold_series is not None:
                        y_pred_val = model_instance.predict(X_val_fold)
                        # Use primary metric from eval config for scoring
                        primary_metric_name = self.config['evaluation']['primary_metric']
                        score_val = self._calculate_metric(y_val_fold_series.to_numpy(), y_pred_val, primary_metric_name)
                        self.model_validation_scores_[model_key] = {
                            primary_metric_name: score_val,
                            'mae': mean_absolute_error(y_val_fold_series.to_numpy(), y_pred_val),
                            'rmse': np.sqrt(mean_squared_error(y_val_fold_series.to_numpy(), y_pred_val))
                        }
                        logging.info(f"ML model {model_key} validation {primary_metric_name}: {score_val:.4f}")
                except Exception as e:
                    logging.error(f"Failed to train ML model {model_key}: {e}")

        # 5. Ensemble Strategy (if applicable)
        # If stacking, base models are already trained. Now train meta-estimator.
        # If voting/averaging, weights might be set based on validation scores.
        ensemble_method = self.config['model_selection']['ensemble_strategy']['method']
        arch_type = self.config['model_selection']['architecture_type']

        if arch_type == 'ensemble':
            if not self.models_:
                raise RuntimeError("Ensemble specified, but no base models were trained successfully.")

            if ensemble_method == 'stacking':
                if self.config['general']['verbose'] > 0: logging.info("Training Stacking Regressor meta-estimator...")
                # Prepare base estimators for StackingRegressor (already trained instances)
                # StackingRegressor internally does CV to generate out-of-fold predictions for meta-learner.
                # Or, if we want to use our already trained models:
                # This requires a custom stacking implementation or careful use of StackingCVRegressor.
                # For simplicity, let's build a new StackingRegressor with untrained base models (configs)
                # and let it handle the fitting. This means re-specifying base models.
                
                stacking_estimators_config = self.config['model_selection']['ml_models'].get(
                    'stacking_regressor', {}).get('params', {}).get('estimators_config', []) # Path to stacker's base models
                
                base_estimators_for_stacking = []
                for est_conf in stacking_estimators_config:
                    model_key = est_conf['key']
                    if model_key in self.config['model_selection']['ml_models']:
                        cfg = self.config['model_selection']['ml_models'][model_key]
                        base_estimators_for_stacking.append((est_conf['name'], self._build_ml_model(model_key, cfg)))
                    elif model_key in self.config['model_selection']['dl_models']:
                        # Keras model in StackingRegressor needs a wrapper (KerasRegressor)
                        # This gets complex if DL models are part of stacking.
                        # For now, assume ML models in StackingRegressor.
                        warn(f"DL model {model_key} in StackingRegressor is not straightforward. Skipping.")

                final_est_key = self.config['model_selection']['ensemble_strategy']['stacking_final_estimator']
                final_est_cfg = self.config['model_selection']['ml_models'].get(final_est_key)
                if final_est_cfg:
                    final_estimator = self._build_ml_model(final_est_key, final_est_cfg)
                else: final_estimator = Ridge() # Default

                stacking_params = {
                    'cv': self.config['model_selection']['ensemble_strategy']['stacking_cv'],
                    'passthrough': self.config['model_selection']['ensemble_strategy']['stacking_passthrough'],
                    'n_jobs': self.config['general']['n_jobs']
                }
                stacker = StackingRegressor(estimators=base_estimators_for_stacking,
                                            final_estimator=final_estimator,
                                            **stacking_params)
                try:
                    stacker.fit(X_train_fold, y_train_fold)
                    self.models_['stacking_ensemble'] = stacker
                    if X_val_fold is not None and y_val_fold_series is not None:
                        y_pred_val_stack = stacker.predict(X_val_fold)
                        primary_metric_name = self.config['evaluation']['primary_metric']
                        score_val_stack = self._calculate_metric(y_val_fold_series.to_numpy(), y_pred_val_stack, primary_metric_name)
                        self.model_validation_scores_['stacking_ensemble'] = {primary_metric_name: score_val_stack}
                        logging.info(f"Stacking Ensemble validation {primary_metric_name}: {score_val_stack:.4f}")
                except Exception as e:
                    logging.error(f"Failed to train Stacking Ensemble: {e}")
            
            elif ensemble_method == 'weighted_average':
                # Determine weights based on validation scores
                # Lower MAE/RMSE is better, Higher R2 is better.
                # Normalize scores and invert if necessary.
                self.model_weights_ = {}
                total_inverse_metric_sum = 0
                primary_metric_name = self.config['evaluation']['primary_metric']
                
                valid_scores_for_weighting = {}
                for model_name, scores_dict in self.model_validation_scores_.items():
                    score = scores_dict.get(primary_metric_name)
                    if score is not None:
                        if primary_metric_name in ['mae', 'rmse'] and score > 1e-9 : # Lower is better, avoid zero
                            valid_scores_for_weighting[model_name] = 1.0 / score
                            total_inverse_metric_sum += (1.0 / score)
                        elif primary_metric_name == 'r2' and score > -float('inf'): # Higher is better
                            # Shift R2 to be positive for weighting (e.g., R2 + 2, if R2 can be very negative)
                            # Or use softmax on R2 values. Simpler: ensure positive.
                            adjusted_score = score + abs(min(0, min(s.get(primary_metric_name,0) for s in self.model_validation_scores_.values() if s.get(primary_metric_name) is not None))) + 1e-6
                            valid_scores_for_weighting[model_name] = adjusted_score
                            total_inverse_metric_sum += adjusted_score
                
                if total_inverse_metric_sum > 1e-9:
                    for model_name, inv_score in valid_scores_for_weighting.items():
                        self.model_weights_[model_name] = inv_score / total_inverse_metric_sum
                else: # Fallback to equal weights
                    active_model_keys = [name for name, scores in self.model_validation_scores_.items() if scores.get(primary_metric_name) is not None]
                    if active_model_keys:
                        equal_weight = 1.0 / len(active_model_keys)
                        self.model_weights_ = {name: equal_weight for name in active_model_keys}
                if self.config['general']['verbose'] > 0: logging.info(f"Ensemble weights (weighted_average): {self.model_weights_}")


        # 6. Select Best Model (if not ensemble or for fallback)
        if self.model_validation_scores_:
            primary_metric = self.config['evaluation']['primary_metric']
            lower_is_better = primary_metric in ['mae', 'rmse']
            
            # Filter out models that might not have the primary metric (e.g., if training failed for one)
            valid_scored_models = {
                name: scores[primary_metric] 
                for name, scores in self.model_validation_scores_.items() 
                if primary_metric in scores and scores[primary_metric] is not None and np.isfinite(scores[primary_metric])
            }
            if valid_scored_models:
                self.best_model_key_ = min(valid_scored_models, key=valid_scored_models.get) if lower_is_better else \
                                    max(valid_scored_models, key=valid_scored_models.get)
                if self.config['general']['verbose'] > 0:
                    logging.info(f"Best single model selected based on validation '{primary_metric}': {self.best_model_key_} "
                                 f"(Score: {valid_scored_models[self.best_model_key_]:.4f})")
            else:
                logging.warning("No valid validation scores found to select best model.")
                if self.models_: self.best_model_key_ = list(self.models_.keys())[0] # Fallback to first model

        elif self.models_: # No validation scores, just pick the first one trained
            self.best_model_key_ = list(self.models_.keys())[0]
            logging.warning(f"No validation scores available. Fallback best model: {self.best_model_key_}")


        # 7. Feature Importances (from ML models or SHAP for DL)
        self._calculate_and_store_feature_importances(X_train_fold, y_train_fold)

        self.is_fitted_ = True
        if self.config['general']['verbose'] > 0: logging.info("`fit` process completed.")
        return self

    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray, metric_name: str) -> float:
        """Helper to calculate a specific metric."""
        if metric_name == 'mae': return mean_absolute_error(y_true, y_pred)
        if metric_name == 'rmse': return np.sqrt(mean_squared_error(y_true, y_pred))
        if metric_name == 'r2': return r2_score(y_true, y_pred)
        # Add other metrics if needed
        raise ValueError(f"Unknown metric for scoring: {metric_name}")

    def _train_keras_model(self, model: Model, model_name: str,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None
                          ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Compiles and trains a Keras model."""
        opt_config = self.config['training']['optimizer']
        opt_type = opt_config['type'].lower()
        lr = opt_config['learning_rate']
        
        optimizer_instance = None
        if opt_type == 'adam': optimizer_instance = Adam(learning_rate=lr, clipnorm=opt_config.get('clipnorm'))
        elif opt_type == 'adamw': optimizer_instance = AdamW(learning_rate=lr, weight_decay=opt_config['weight_decay'], clipnorm=opt_config.get('clipnorm'))
        elif opt_type == 'nadam': optimizer_instance = Nadam(learning_rate=lr, clipnorm=opt_config.get('clipnorm'))
        elif opt_type == 'rmsprop': optimizer_instance = RMSprop(learning_rate=lr, clipnorm=opt_config.get('clipnorm'))
        else: raise ValueError(f"Unsupported optimizer type: {opt_type}")

        model.compile(optimizer=optimizer_instance,
                      loss=self.config['training']['loss_function'],
                      metrics=self.config['training']['metrics'])

        callbacks = []
        cb_configs = self.config['training']['callbacks']
        if cb_configs['early_stopping']['active']:
            callbacks.append(EarlyStopping(
                monitor=cb_configs['early_stopping']['monitor'],
                patience=cb_configs['early_stopping']['patience'],
                restore_best_weights=cb_configs['early_stopping']['restore_best_weights'],
                verbose=self.config['general']['verbose']
            ))
        if cb_configs['reduce_lr_on_plateau']['active']:
            callbacks.append(ReduceLROnPlateau(
                monitor=cb_configs['reduce_lr_on_plateau']['monitor'],
                factor=cb_configs['reduce_lr_on_plateau']['factor'],
                patience=cb_configs['reduce_lr_on_plateau']['patience'],
                min_lr=cb_configs['reduce_lr_on_plateau']['min_lr'],
                verbose=self.config['general']['verbose']
            ))
        if cb_configs['model_checkpoint']['active']:
            results_dir = Path(self.config['general']['results_dir']) / self.biomarker_type / "checkpoints"
            results_dir.mkdir(parents=True, exist_ok=True)
            filepath = results_dir / cb_configs['model_checkpoint']['filename_template'].format(model_name=model_name, biomarker=self.biomarker_type)
            callbacks.append(ModelCheckpoint(
                filepath=str(filepath),
                monitor=cb_configs['model_checkpoint']['monitor'],
                save_best_only=cb_configs['model_checkpoint']['save_best_only'],
                save_weights_only=False, # Save full model for easier loading
                verbose=self.config['general']['verbose']
            ))
        
        # Data Augmentation (placeholder, needs specific implementation if active)
        # if self.config['training']['data_augmentation']['active']:
        #     # X_train, y_train = self._augment_data(X_train, y_train, self.config['training']['data_augmentation']['methods'])
        #     pass

        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = None
        try:
            history_obj = model.fit(
                X_train, y_train,
                batch_size=self.config['training']['batch_size'],
                epochs=self.config['training']['epochs'],
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=self.config['general']['verbose']
            )
            history = history_obj.history
        except Exception as e:
            logging.error(f"Error training Keras model {model_name}: {e}")
            return None, None # Return None if training fails

        val_metrics_output = None
        if validation_data and history:
            # Get best validation score based on early stopping monitor
            monitor_metric = cb_configs['early_stopping']['monitor']
            if monitor_metric in history:
                best_val_score = min(history[monitor_metric]) if 'loss' in monitor_metric or 'mae' in monitor_metric or 'mse' in monitor_metric else \
                                 max(history[monitor_metric]) # e.g. for r2 or accuracy
                
                val_metrics_output = {
                    self.config['evaluation']['primary_metric']: best_val_score, # This assumes monitor metric is primary
                    'val_loss_best': min(history.get('val_loss', [np.inf])),
                    'val_mae_best': min(history.get('val_mae', [np.inf])),
                    'val_mse_best': min(history.get('val_mse', [np.inf])),
                }
                # Ensure primary metric is correctly populated if different from monitor
                primary_eval_metric = self.config['evaluation']['primary_metric']
                if f'val_{primary_eval_metric}' in history:
                     val_metrics_output[primary_eval_metric] = min(history[f'val_{primary_eval_metric}']) if primary_eval_metric in ['mae','mse','loss'] else max(history[f'val_{primary_eval_metric}'])

                if self.config['general']['verbose'] > 0:
                    logging.info(f"Keras model {model_name} best validation '{monitor_metric}': {best_val_score:.4f}")
            else:
                 logging.warning(f"Monitor metric '{monitor_metric}' not found in Keras history for {model_name}.")
        return history, val_metrics_output

    def _run_hpo_ray_tune(self, X_train_full: pd.DataFrame, y_train_full: pd.Series,
                          X_val_hpo: Optional[pd.DataFrame] = None, y_val_hpo: Optional[pd.Series] = None,
                          groups_hpo: Optional[pd.Series] = None) -> Optional[Dict]:
        """Runs Hyperparameter Optimization using Ray Tune."""
        if not has_ray_tune:
            logging.warning("Ray Tune not installed. Skipping HPO.")
            return None

        hpo_cfg = self.config['hyperparameter_optimization']['ray_tune_config']
        
        # Preprocess data for HPO (once, outside the tune.run loop for efficiency)
        # HPO trials will receive already processed data.
        if self.preprocessor_pipeline_ is None: # Should be initialized if called from fit after preprocessor init
            self.preprocessor_pipeline_ = self._initialize_preprocessor_pipeline()

        if self.preprocessor_pipeline_:
            # Important: fit_transform on full training data for HPO, then split within trial or use provided val set
            logging.info("HPO: Applying preprocessor to full training data...")
            X_processed_hpo = self.preprocessor_pipeline_.fit_transform(X_train_full, y_train_full)
            if X_val_hpo is not None:
                X_val_processed_hpo = self.preprocessor_pipeline_.transform(X_val_hpo)
            else: X_val_processed_hpo = None
        else:
            X_processed_hpo = X_train_full.copy()
            X_val_processed_hpo = X_val_hpo.copy() if X_val_hpo is not None else None

        if isinstance(X_processed_hpo, pd.DataFrame): X_processed_hpo_np = X_processed_hpo.to_numpy()
        else: X_processed_hpo_np = X_processed_hpo
        y_train_full_np = y_train_full.to_numpy()

        if X_val_processed_hpo is not None:
            if isinstance(X_val_processed_hpo, pd.DataFrame): X_val_processed_hpo_np = X_val_processed_hpo.to_numpy()
            else: X_val_processed_hpo_np = X_val_processed_hpo
            y_val_hpo_np = y_val_hpo.to_numpy() if y_val_hpo is not None else None
        else: # Need to split inside trial if no val set provided
            X_val_processed_hpo_np = None
            y_val_hpo_np = None


        # Define search space and trainable for Ray Tune
        # This example focuses on HPO for a single primary DL model.
        # HPO for ensembles or multiple model types is more complex.
        primary_dl_model_name = self.config['model_selection'].get('primary_model_name_for_hpo',
                                                                  self.config['model_selection']['primary_model_name'])
        
        if primary_dl_model_name not in self.config['model_selection']['dl_models']:
            logging.warning(f"Primary DL model for HPO '{primary_dl_model_name}' not in DL model configs. Skipping HPO.")
            return None
            
        model_hpo_config = self.config['model_selection']['dl_models'][primary_dl_model_name]
        
        # Construct search space
        search_space = {}
        # Common DL params
        common_dl_hpo_params = hpo_cfg.get("param_space_common_dl", {})
        for p_name, p_config in common_dl_hpo_params.items():
            if p_config['type'] == 'loguniform': search_space[p_name] = tune.loguniform(p_config['lower'], p_config['upper'])
            elif p_config['type'] == 'uniform': search_space[p_name] = tune.uniform(p_config['lower'], p_config['upper'])
            elif p_config['type'] == 'choice': search_space[p_name] = tune.choice(p_config['values'])
            elif p_config['type'] == 'randint': search_space[p_name] = tune.randint(p_config['lower'], p_config['upper'])
        
        # Model-specific params (e.g., num_filters, rnn_units)
        # This needs to be defined in the config, e.g., under model_hpo_config['hpo_params']
        model_specific_hpo_params = model_hpo_config.get("hpo_params", {})
        for p_name, p_config in model_specific_hpo_params.items():
             # Similar logic to populate search_space based on p_config['type']
             if p_config['type'] == 'choice': search_space[p_name] = tune.choice(p_config['values'])
             # ... add other types
        
        if not search_space:
            logging.warning("HPO search space is empty. Check HPO configuration. Skipping HPO.")
            return None

        # Data to be used by each trial (must be Ray object store or accessible)
        # For smaller datasets, can pass directly. For larger, use ray.put()
        X_train_ref = ray.put(X_processed_hpo_np)
        y_train_ref = ray.put(y_train_full_np)
        X_val_ref = ray.put(X_val_processed_hpo_np) if X_val_processed_hpo_np is not None else None
        y_val_ref = ray.put(y_val_hpo_np) if y_val_hpo_np is not None else None
        
        # Get input shape for model builder from processed HPO data
        hpo_dl_input_shape = X_processed_hpo_np.shape[1:]
        hpo_num_outputs = 1

        # Trainable function for Ray Tune
        # `hpo_trial_config` will contain sampled hyperparameters for the current trial
        def hpo_trainable(hpo_trial_config: Dict):
            tf.keras.backend.clear_session() # Ensure clean state for each trial
            
            # Get data for this trial
            # Note: ray.get() can be slow if data is large and not efficiently shared.
            # Consider `ray.train.DataParallelTrainer` for distributed data loading/training.
            current_X_train = ray.get(X_train_ref)
            current_y_train = ray.get(y_train_ref)
            current_X_val = ray.get(X_val_ref) if X_val_ref is not None else None
            current_y_val = ray.get(y_val_ref) if y_val_ref is not None else None

            # If no validation set was provided for HPO, split train data here
            if current_X_val is None or current_y_val is None:
                train_idx, val_idx = train_test_split(
                    np.arange(len(current_X_train)),
                    test_size=self.config['training']['validation_split'], # Use main config's split
                    random_state=self.config['general']['random_state'], # Consistent split
                    shuffle=True
                )
                trial_X_train, trial_X_val = current_X_train[train_idx], current_X_train[val_idx]
                trial_y_train, trial_y_val = current_y_train[train_idx], current_y_train[val_idx]
            else: # Use the provided HPO validation set
                trial_X_train, trial_y_train = current_X_train, current_y_train
                trial_X_val, trial_y_val = current_X_val, current_y_val

            # Build model with hyperparameters from hpo_trial_config
            # Merge hpo_trial_config into the base model config
            trial_model_base_config = model_hpo_config['config'].copy() # Start with default model config
            # Update with common DL HPO params (like learning_rate, batch_size)
            for common_param in common_dl_hpo_params.keys():
                if common_param == 'learning_rate': # Update optimizer config
                    trial_model_base_config.setdefault('optimizer', {}).update({'learning_rate': hpo_trial_config[common_param]})
                elif common_param == 'batch_size': # This is used in fit, not model build
                    pass 
                else: # Other common params might be directly in model config (e.g. dropout)
                    trial_model_base_config[common_param] = hpo_trial_config[common_param]
            
            # Update with model-specific HPO params
            for specific_param in model_specific_hpo_params.keys():
                trial_model_base_config[specific_param] = hpo_trial_config[specific_param]

            # Create a temporary config for this trial's model builder using the sampled HPs
            temp_full_dl_config_for_trial = {'builder': model_hpo_config['builder'], 'config': trial_model_base_config}
            
            trial_model = self._build_dl_model(f"{primary_dl_model_name}_hpo", temp_full_dl_config_for_trial,
                                               hpo_dl_input_shape, hpo_num_outputs)
            
            # Compile model (optimizer learning rate is from hpo_trial_config)
            trial_opt_config = self.config['training']['optimizer'].copy()
            trial_opt_config['learning_rate'] = hpo_trial_config.get('learning_rate', trial_opt_config['learning_rate'])
            
            opt_type = trial_opt_config['type'].lower()
            lr = trial_opt_config['learning_rate']
            optimizer_instance_trial = None
            if opt_type == 'adam': optimizer_instance_trial = Adam(learning_rate=lr, clipnorm=trial_opt_config.get('clipnorm'))
            elif opt_type == 'adamw': optimizer_instance_trial = AdamW(learning_rate=lr, weight_decay=trial_opt_config['weight_decay'], clipnorm=trial_opt_config.get('clipnorm'))
            # ... add other optimizers
            else: optimizer_instance_trial = Adam(learning_rate=lr) # Default

            trial_model.compile(optimizer=optimizer_instance_trial,
                                loss=self.config['training']['loss_function'],
                                metrics=self.config['training']['metrics'])
            
            # Callbacks for Tune (reporting metrics)
            trial_callbacks = [TuneReportCallback({
                hpo_cfg["metric_to_optimize"]: hpo_cfg["metric_to_optimize"] # e.g. "val_mae": "val_mae"
            })]
            # Add early stopping for trials if desired (from main config)
            main_cb_configs = self.config['training']['callbacks']
            if main_cb_configs['early_stopping']['active']:
                trial_callbacks.append(EarlyStopping(
                    monitor=main_cb_configs['early_stopping']['monitor'], # Usually same as metric_to_optimize
                    patience=main_cb_configs['early_stopping']['patience'] // 2, # Shorter patience for HPO trials
                    restore_best_weights=False # Not strictly needed for HPO, final model retrained
                ))

            trial_batch_size = hpo_trial_config.get('batch_size', self.config['training']['batch_size'])
            
            trial_model.fit(trial_X_train, trial_y_train,
                            batch_size=trial_batch_size,
                            epochs=self.config['training']['epochs'], # Can also be an HPO parameter
                            validation_data=(trial_X_val, trial_y_val),
                            callbacks=trial_callbacks,
                            verbose=0) # Usually silent during HPO trials

        # Scheduler
        scheduler = None
        if hpo_cfg['scheduler'] == 'asha':
            scheduler = ASHAScheduler(metric=hpo_cfg["metric_to_optimize"], mode=hpo_cfg["optimization_mode"],
                                      max_t=self.config['training']['epochs'], grace_period=10, reduction_factor=2)
        elif hpo_cfg['scheduler'] == 'pbt':
            scheduler = PopulationBasedTraining(metric=hpo_cfg["metric_to_optimize"], mode=hpo_cfg["optimization_mode"],
                                                # hyperparam_mutations needs to be defined based on search_space
                                                # PBT is more complex to set up correctly.
                                                time_attr='training_iteration') # Or 'time_total_s'
            logging.warning("PBT scheduler for Ray Tune requires careful setup of hyperparam_mutations. Using ASHA as fallback if not well-configured.")
            scheduler = ASHAScheduler(metric=hpo_cfg["metric_to_optimize"], mode=hpo_cfg["optimization_mode"]) # Fallback
        # ... add other schedulers like HyperBand

        # Search Algorithm
        search_alg = None
        if hpo_cfg['search_alg'] == 'hyperopt':
            search_alg = HyperOptSearch(metric=hpo_cfg["metric_to_optimize"], mode=hpo_cfg["optimization_mode"])
        elif hpo_cfg['search_alg'] == 'optuna':
            search_alg = OptunaSearch(metric=hpo_cfg["metric_to_optimize"], mode=hpo_cfg["optimization_mode"])
        # ... add other search algorithms like BOHB, RandomSearch

        # Set up Ray Tune execution
        if not ray.is_initialized():
            try:
                ray.init(num_cpus=os.cpu_count(), num_gpus=len(tf.config.list_physical_devices('GPU')),
                         ignore_reinit_error=True, logging_level=logging.ERROR) # Suppress Ray's own INFO logs
            except Exception as e:
                logging.error(f"Ray initialization failed: {e}. Cannot run HPO.")
                return None
        
        analysis = None
        try:
            reporter = tune.CLIReporter(metric_columns=[hpo_cfg["metric_to_optimize"], "training_iteration"])
            analysis = tune.run(
                hpo_trainable,
                config=search_space,
                name=f"{self.biomarker_type}_{primary_dl_model_name}_hpo_run",
                scheduler=scheduler,
                search_alg=search_alg,
                num_samples=hpo_cfg['num_samples'],
                resources_per_trial=hpo_cfg['resources_per_trial'],
                progress_reporter=reporter,
                local_dir=str(Path(self.config['general']['results_dir']) / self.biomarker_type / "hpo_ray_results"),
                verbose=1 if self.config['general']['verbose'] > 0 else 0, # Tune's own verbosity
                stop={"training_iteration": self.config['training']['epochs']} # Alternative stopping condition
            )
        except Exception as e:
            logging.error(f"Ray Tune `tune.run` failed: {e}")
            if ray.is_initialized(): ray.shutdown()
            return None

        if ray.is_initialized(): ray.shutdown() # Clean up Ray

        if analysis:
            best_trial = analysis.get_best_trial(metric=hpo_cfg["metric_to_optimize"],
                                                 mode=hpo_cfg["optimization_mode"],
                                                 scope="all") # Get best trial over all (even stopped ones)
            if best_trial:
                logging.info(f"Best HPO trial ID: {best_trial.trial_id}")
                logging.info(f"Best HPO trial config: {best_trial.config}")
                logging.info(f"Best HPO trial {hpo_cfg['metric_to_optimize']}: {best_trial.last_result[hpo_cfg['metric_to_optimize']]:.4f}")
                return best_trial.config # Return the best hyperparameter configuration
            else:
                logging.warning("Ray Tune analysis did not yield a best trial.")
        return None

    def _calculate_and_store_feature_importances(self, X_train_for_fi: np.ndarray, y_train_for_fi: np.ndarray):
        """Calculates and stores feature importances."""
        if self.feature_names_processed_ is None:
            logging.warning("Processed feature names not available, cannot calculate feature importances.")
            return

        importances_df_list = []
        # For ML models with built-in feature importance
        for model_name, model_instance in self.models_.items():
            if hasattr(model_instance, 'feature_importances_'): # Tree-based
                fi = model_instance.feature_importances_
                df = pd.DataFrame({'feature': self.feature_names_processed_, 'importance': fi, 'model': model_name})
                importances_df_list.append(df)
            elif hasattr(model_instance, 'coef_'): # Linear models
                if model_instance.coef_.ndim == 1: # Simple linear
                    fi = np.abs(model_instance.coef_)
                elif model_instance.coef_.ndim == 2 and model_instance.coef_.shape[0] == 1: # e.g. SVR with linear kernel
                    fi = np.abs(model_instance.coef_.flatten())
                else: # Multi-output or complex coef structure
                    fi = np.mean(np.abs(model_instance.coef_), axis=0) if model_instance.coef_.ndim > 1 else np.abs(model_instance.coef_)
                
                if len(fi) == len(self.feature_names_processed_):
                    df = pd.DataFrame({'feature': self.feature_names_processed_, 'importance': fi, 'model': model_name})
                    importances_df_list.append(df)
                else:
                    logging.warning(f"Mismatch in coef length ({len(fi)}) and feature names ({len(self.feature_names_processed_)}) for {model_name}")

        # For DL models, Permutation Importance or SHAP would be needed (complex to integrate here generically)
        # Placeholder for SHAP:
        # if any(isinstance(m, tf.keras.Model) for m in self.models_.values()):
        #    logging.info("Feature importance for DL models requires SHAP or PermutationImportance (not auto-calculated here).")

        if importances_df_list:
            full_importances_df = pd.concat(importances_df_list)
            # Aggregate (e.g., mean importance across models)
            self.feature_importances_ = full_importances_df.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
            if self.config['general']['verbose'] > 0:
                logging.info(f"Top 10 aggregated feature importances:\n{self.feature_importances_.head(10)}")
        else:
            self.feature_importances_ = None
            logging.info("No feature importances could be extracted from the trained models.")
    def predict(self, X: Union[pd.DataFrame, np.ndarray],
                return_uncertainty: bool = False,
                use_best_single_model: bool = False # If True, overrides ensemble to use best_model_key_
               ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Makes predictions on new data.
        Handles preprocessing and uses the fitted model(s).

        Args:
            X: Input data (DataFrame or NumPy array).
            return_uncertainty: If True, also returns uncertainty estimates.
            use_best_single_model: If True, forces prediction using only the `best_model_key_`.

        Returns:
            Predictions (np.ndarray). If return_uncertainty is True, returns (predictions, uncertainties).
        """
        if not self.is_fitted_:
            raise NotFittedError("Predictor is not fitted yet. Call 'fit' first.")
        if self.config['general']['verbose'] > 0: logging.info("Starting prediction...")

        if isinstance(X, np.ndarray):
            if self.feature_names_in_ is None:
                # This case should ideally not happen if fit was called on a DataFrame first.
                # If X is numpy and feature_names_in_ is None, assume columns are f_0, f_1, ...
                # Or require X to be DataFrame if feature_names_in_ is None.
                raise ValueError("Predictor was likely fitted on an ndarray without feature names. Please provide X as DataFrame or ensure feature_names_in_ is set.")
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            raise TypeError("Input X must be a pandas DataFrame or NumPy array.")

        # Preprocess data using the fitted pipeline
        if self.preprocessor_pipeline_:
            if self.config['general']['verbose'] > 1: logging.debug("Applying preprocessing pipeline to prediction data...")
            X_processed = self.preprocessor_pipeline_.transform(X_df)
        else:
            X_processed = X_df
        
        if isinstance(X_processed, pd.DataFrame): X_processed_np = X_processed.to_numpy()
        else: X_processed_np = X_processed # Assume already numpy

        # Determine which model(s) to use for prediction
        predictions_dict = {}
        uncertainties_dict = {} # For methods like MC Dropout per model

        models_to_predict_with = {}
        arch_type = self.config['model_selection']['architecture_type']

        if use_best_single_model or arch_type in ['single_dl', 'single_ml']:
            if not self.best_model_key_ or self.best_model_key_ not in self.models_:
                raise RuntimeError(f"Best model '{self.best_model_key_}' not found or not trained.")
            models_to_predict_with[self.best_model_key_] = self.models_[self.best_model_key_]
        elif arch_type == 'ensemble':
            ensemble_method = self.config['model_selection']['ensemble_strategy']['method']
            if ensemble_method == 'stacking' and 'stacking_ensemble' in self.models_:
                # Stacking regressor handles internal predictions from base models
                models_to_predict_with['stacking_ensemble'] = self.models_['stacking_ensemble']
            else: # For voting, weighted_average, or if stacking model failed, use all base models
                models_to_predict_with = self.models_.copy()
                if 'stacking_ensemble' in models_to_predict_with: # Don't use stacker as a base model for another ensemble
                    del models_to_predict_with['stacking_ensemble']
        else: # Should not happen
            raise ValueError(f"Invalid architecture_type '{arch_type}' for prediction.")

        if not models_to_predict_with:
             raise RuntimeError("No models available for prediction.")


        for model_name, model_instance in models_to_predict_with.items():
            if self.config['general']['verbose'] > 1: logging.debug(f"Predicting with model: {model_name}")
            if isinstance(model_instance, tf.keras.Model):
                if return_uncertainty and self.config['uncertainty_estimation']['method'] == 'mc_dropout':
                    # Enable dropout layers for MC Dropout (if they exist)
                    mc_predictions = []
                    num_mc_samples = self.config['uncertainty_estimation']['mc_dropout_samples']
                    for _ in range(num_mc_samples):
                        # Keras models need training=True for dropout to be active during inference
                        try:
                            # Check if model supports 'training' argument in call
                            if 'training' in inspect.signature(model_instance.call).parameters:
                                pred = model_instance(X_processed_np, training=True).numpy()
                            else: # Some models might not have dropout or explicit training arg in call
                                pred = model_instance.predict(X_processed_np)
                        except TypeError: # Fallback if 'training' arg is not expected by predict
                             pred = model_instance.predict(X_processed_np)

                        mc_predictions.append(pred.flatten())
                    
                    mc_predictions_np = np.array(mc_predictions) # Shape (num_mc_samples, num_data_points)
                    predictions_dict[model_name] = np.mean(mc_predictions_np, axis=0)
                    uncertainties_dict[model_name] = np.std(mc_predictions_np, axis=0)
                else: # Standard prediction for Keras model
                    predictions_dict[model_name] = model_instance.predict(X_processed_np).flatten()
            else: # Scikit-learn model
                predictions_dict[model_name] = model_instance.predict(X_processed_np).flatten()
                # For sklearn, uncertainty might come from predict_proba (if classifier) or quantile regressors.
                # For regressors, if it's a BayesianRidge or similar, it might have std.
                if return_uncertainty and hasattr(model_instance, 'predict') and 'return_std' in inspect.signature(model_instance.predict).parameters:
                    try:
                        _, std_dev = model_instance.predict(X_processed_np, return_std=True)
                        uncertainties_dict[model_name] = std_dev.flatten()
                    except TypeError: pass # Model doesn't support return_std

        # Combine predictions if multiple models were used (e.g., for ensemble_variance or weighted_average)
        if len(predictions_dict) == 1: # Single model prediction (best_model, single_dl/ml, or stacker)
            final_predictions = list(predictions_dict.values())[0]
            final_uncertainties = list(uncertainties_dict.values())[0] if uncertainties_dict and return_uncertainty else None
        elif arch_type == 'ensemble' and self.config['model_selection']['ensemble_strategy']['method'] == 'weighted_average':
            if not self.model_weights_:
                logging.warning("Weighted average ensemble selected, but no model weights found. Using equal weights.")
                num_models_in_pred = len(predictions_dict)
                weights_to_use = {name: 1.0/num_models_in_pred for name in predictions_dict.keys()}
            else:
                weights_to_use = self.model_weights_
            
            weighted_preds_sum = np.zeros_like(list(predictions_dict.values())[0])
            sum_of_weights = 0
            for model_name, preds in predictions_dict.items():
                weight = weights_to_use.get(model_name, 0) # Get weight, default to 0 if model not in weights dict
                if weight > 0 : # Only include models with positive weight
                    weighted_preds_sum += weight * preds
                    sum_of_weights += weight
            final_predictions = weighted_preds_sum / sum_of_weights if sum_of_weights > 0 else np.mean(list(predictions_dict.values()), axis=0)
            
            # Uncertainty for weighted average could be weighted average of variances, or overall variance
            if return_uncertainty and self.config['uncertainty_estimation']['method'] == 'ensemble_variance':
                 all_preds_for_variance = np.array(list(predictions_dict.values())) # (num_models, num_data_points)
                 final_uncertainties = np.std(all_preds_for_variance, axis=0)
            elif return_uncertainty and uncertainties_dict: # If individual uncertainties exist
                # Weighted average of variances (std^2)
                weighted_vars_sum = np.zeros_like(final_predictions)
                sum_of_weights_var = 0
                for model_name, std_devs in uncertainties_dict.items():
                    weight = weights_to_use.get(model_name, 0)
                    if weight > 0:
                        weighted_vars_sum += weight * (std_devs**2)
                        sum_of_weights_var += weight
                final_uncertainties = np.sqrt(weighted_vars_sum / sum_of_weights_var) if sum_of_weights_var > 0 else None
            else:
                final_uncertainties = None
        else: # Default to simple average if other ensemble logic not fully matched
            logging.info("Using simple average for ensemble predictions.")
            all_preds_for_avg = np.array(list(predictions_dict.values()))
            final_predictions = np.mean(all_preds_for_avg, axis=0)
            if return_uncertainty and self.config['uncertainty_estimation']['method'] == 'ensemble_variance':
                final_uncertainties = np.std(all_preds_for_avg, axis=0)
            else:
                final_uncertainties = None # Or other methods if available

        if self.config['general']['verbose'] > 0: logging.info("Prediction completed.")
        if return_uncertainty:
            if final_uncertainties is None: # Fallback uncertainty if not calculated
                final_uncertainties = np.full_like(final_predictions, np.nan)
                logging.warning("Requested uncertainty but no method yielded results. Returning NaNs for uncertainty.")
            return final_predictions, final_uncertainties
        return final_predictions

    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y_true: Union[pd.Series, np.ndarray],
                 use_best_single_model: bool = False) -> Dict[str, float]:
        """Evaluates the predictor on given test data."""
        if not self.is_fitted_:
            raise NotFittedError("Predictor is not fitted yet. Call 'fit' first.")
        if self.config['general']['verbose'] > 0: logging.info(f"Evaluating predictor for {self.biomarker_type}...")

        y_pred = self.predict(X, return_uncertainty=False, use_best_single_model=use_best_single_model)
        
        if isinstance(y_true, pd.Series): y_true_np = y_true.to_numpy()
        else: y_true_np = np.asarray(y_true)

        metrics = {}
        metrics['mae'] = mean_absolute_error(y_true_np, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true_np, y_pred))
        metrics['r2'] = r2_score(y_true_np, y_pred)
        # MAPE (handle potential zeros in y_true)
        abs_error_ratio = np.abs((y_true_np - y_pred) / (y_true_np + 1e-9)) # Add epsilon to avoid div by zero
        metrics['mape'] = np.mean(abs_error_ratio[np.isfinite(abs_error_ratio)]) * 100

        if self.biomarker_type == 'glucose' and self.config['evaluation']['clarke_error_grid_for_glucose']:
            ceg_results = self._calculate_clarke_error_grid(y_true_np, y_pred)
            metrics.update(ceg_results)
        
        if self.biomarker_type == 'spo2':
            for threshold_pct in self.config['evaluation']['spo2_accuracy_thresholds_percent']:
                # SpO2 is usually 0-1 or 0-100. Assuming 0-1 from model output.
                # Threshold is percent, so 2.0 means 0.02 in 0-1 scale.
                accuracy_within_thresh = np.mean(np.abs(y_true_np - y_pred) <= (threshold_pct / 100.0)) * 100
                metrics[f'spo2_accuracy_within_{threshold_pct}pct'] = accuracy_within_thresh
        
        if self.config['general']['verbose'] > 0:
            log_msg = f"Evaluation metrics for {self.biomarker_type}: "
            for k,v in metrics.items(): log_msg += f"{k}={v:.4f}, "
            logging.info(log_msg.strip(", "))
        return metrics

    def _calculate_clarke_error_grid(self, y_true_mg_dl: np.ndarray, y_pred_mg_dl: np.ndarray) -> Dict[str, float]:
        """Calculates Clarke Error Grid Analysis percentages for glucose."""
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true_mg_dl)
        y_pred = np.asarray(y_pred_mg_dl)

        points_in_zone = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
        total_points = len(y_true)
        if total_points == 0: return {f'clarke_zone_{z}': 0.0 for z in points_in_zone}

        for i in range(total_points):
            gt, pred = y_true[i], y_pred[i]
            zone = ''
            if (gt < 70 and pred < 70) or \
               (gt >= 70 and (abs(gt - pred) / gt * 100) <= 20) or \
               (gt >= 70 and abs(gt - pred) <= (0.2 * gt)): # Zone A (accurate)
                zone = 'A'
            elif (gt >= 70 and pred >= 70 and ((gt >= 180 and pred <= 70) or (gt <= 70 and pred >= 180))): # Zone E (erroneous, opposite)
                 zone = 'E' # This condition is tricky and covered by others too. Redo CEG logic.

            # Simplified standard CEG logic:
            if (pred >= 70 and (pred <= 0.8 * gt + 56)) or \
               (pred < 70 and gt < 70) or \
               (gt >= 70 and pred >= 0.8 * gt - 14 and pred <= 1.2 * gt + 14 and not (gt >= 180 and pred <=70) and not (gt<=70 and pred >= 180)): # Zone A
                 # More precise Zone A: within 20% or if both < 70 and within certain boundary
                 if (abs(gt-pred) <= 0.2*gt) or (gt < 70 and pred < 70) or \
                    (gt >= 70 and pred >= 70 and abs(gt-pred) <= 0.2*min(gt,pred)): # Try another A def
                    points_in_zone['A'] +=1; continue

            # This CEG logic needs to be robustly implemented based on standard definitions.
            # The Parkes Error Grid definition is often used.
            # For brevity here, I'm providing a conceptual placeholder.
            # A proper CEG implementation requires careful boundary definitions.
            # Using a simplified heuristic based on relative error for now.
            # THIS IS A SIMPLIFICATION AND NOT A VALID CEG.
            # A real CEG is a series of boundary lines.
            # Example of finding which zone a point (x,y) (true,pred) falls into:
            # if y > x and y < (7/5)*x + (gt>130? -182: 58.5): B_upper etc.
            # Due to complexity, this is a placeholder.
            # Let's use a very simplified criteria:
            rel_err = abs(gt - pred) / (gt + 1e-6)
            abs_err = abs(gt - pred)

            if rel_err <= 0.15 or abs_err <= 15: points_in_zone['A'] += 1
            elif rel_err <= 0.30 or abs_err <= 30: points_in_zone['B'] += 1
            elif gt < 70 and pred > 180: points_in_zone['D'] +=1 # Missed hypo
            elif gt > 240 and pred < 70: points_in_zone['D'] +=1 # Missed hyper
            elif (gt > 70 and pred < 40) or (gt < 180 and pred > 300): points_in_zone['C'] +=1
            else: points_in_zone['E'] +=1 # Catch-all for very wrong

        warn("Clarke Error Grid logic used is highly simplified and not a standard implementation. For clinical relevance, use a validated CEG library or implement full Parkes criteria.")
        return {f'clarke_zone_{z}': (count / total_points) * 100.0 for z, count in points_in_zone.items()}

    def save(self, directory_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves the fitted predictor (config, preprocessor, models) to a directory."""
        if not self.is_fitted_:
            raise NotFittedError("Predictor is not fitted. Nothing to save.")

        if directory_path is None:
            base_dir = Path(self.config['general']['results_dir']) / self.biomarker_type / "saved_predictors"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = base_dir / f"{self.biomarker_type}_predictor_{timestamp}"
        else:
            save_dir = Path(directory_path)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        if self.config['general']['verbose'] > 0: logging.info(f"Saving predictor to: {save_dir}")

        # 1. Save configuration (cleaned for serialization)
        config_to_save = self._clean_config_for_serialization(self.config)
        with open(save_dir / "predictor_config.json", 'w') as f:
            json.dump(config_to_save, f, indent=4)

        # 2. Save preprocessor pipeline
        if self.preprocessor_pipeline_:
            joblib.dump(self.preprocessor_pipeline_, save_dir / "preprocessor_pipeline.joblib",
                        compress=self.config['saving_loading']['compress_joblib'])

        # 3. Save models
        models_save_dir = save_dir / "models"
        models_save_dir.mkdir(exist_ok=True)
        for model_name, model_instance in self.models_.items():
            if isinstance(model_instance, tf.keras.Model):
                model_instance.save(models_save_dir / f"{model_name}_keras_model", save_format="tf")
            else: # Scikit-learn model
                joblib.dump(model_instance, models_save_dir / f"{model_name}_sklearn_model.joblib",
                            compress=self.config['saving_loading']['compress_joblib'])
        
        # 4. Save other attributes
        metadata = {
            "biomarker_type": self.biomarker_type,
            "is_fitted": self.is_fitted_,
            "best_model_key": self.best_model_key_,
            "model_weights": self.model_weights_ if self.model_weights_ else None,
            "feature_names_in": self.feature_names_in_,
            "feature_names_processed": self.feature_names_processed_,
            "model_validation_scores": self.model_validation_scores_,
            "training_history_keys": list(self.training_history_.keys()), # Save keys, history can be large
            "python_version": sys.version,
            "tensorflow_version": tf.__version__,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "save_timestamp": datetime.now().isoformat()
        }
        with open(save_dir / "predictor_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)

        # Save feature importances if available
        if self.feature_importances_ is not None:
            self.feature_importances_.to_csv(save_dir / "feature_importances.csv", index=False)
        
        # Optionally save training history (can be large)
        # For now, only keys are saved in metadata. Full history saving could be added.

        if self.config['general']['verbose'] > 0: logging.info("Predictor saved successfully.")
        return save_dir

    @classmethod
    def load(cls, directory_path: Union[str, Path]) -> 'PhysicsInformedBiomarkerPredictor':
        """Loads a predictor from a saved directory."""
        load_dir = Path(directory_path)
        if not load_dir.exists() or not load_dir.is_dir():
            raise FileNotFoundError(f"Saved predictor directory not found: {load_dir}")
        
        logging.info(f"Loading predictor from: {load_dir}")

        # 1. Load metadata and config
        with open(load_dir / "predictor_metadata.json", 'r') as f:
            metadata = json.load(f)
        with open(load_dir / "predictor_config.json", 'r') as f:
            config = json.load(f) # Config is already cleaned

        # Create instance (biomarker_type from metadata, config from file)
        # Need to handle templated values in config if they were saved as such (unlikely with current save)
        instance = cls(biomarker_type=metadata['biomarker_type'], config=config)
        
        # Restore attributes from metadata
        instance.is_fitted_ = metadata.get('is_fitted', False)
        instance.best_model_key_ = metadata.get('best_model_key')
        instance.model_weights_ = metadata.get('model_weights')
        instance.feature_names_in_ = metadata.get('feature_names_in')
        instance.feature_names_processed_ = metadata.get('feature_names_processed')
        instance.model_validation_scores_ = metadata.get('model_validation_scores', {})
        # Training history not fully restored, only keys mentioned.

        # 2. Load preprocessor pipeline
        preprocessor_path = load_dir / "preprocessor_pipeline.joblib"
        if preprocessor_path.exists():
            instance.preprocessor_pipeline_ = joblib.load(preprocessor_path)
        else:
            instance.preprocessor_pipeline_ = None # Or re-initialize if config allows

        # 3. Load models
        models_load_dir = load_dir / "models"
        instance.models_ = {}
        if models_load_dir.exists():
            for model_file_or_dir in models_load_dir.iterdir():
                model_name = ""
                loaded_model = None
                if model_file_or_dir.is_dir() and model_file_or_dir.name.endswith("_keras_model"): # TF SavedModel format
                    model_name = model_file_or_dir.name.replace("_keras_model", "")
                    # Need to provide custom objects if physics layers were used
                    custom_objects = {
                        "BeerLambertLawLayer": BeerLambertLawLayer,
                        "ScatteringModelLayer": ScatteringModelLayer,
                        "PhysiologicalConstraintLayer": PhysiologicalConstraintLayer,
                        "ResidualConv1DBlock": ResidualConv1DBlock,
                        "AttentionBlock": AttentionBlock,
                        # Add any other custom layers/activations used by saved models
                        "gelu": tf.nn.gelu, # If GELU was used as string in config
                        "LeakyReLU": LeakyReLU # If used via string config
                    }
                    try:
                        loaded_model = tf.keras.models.load_model(model_file_or_dir, custom_objects=custom_objects, compile=False) # Recompile manually if needed
                        # Manually recompile if optimizer state not critical or needs adjustment
                        # This part is tricky if optimizer params changed.
                        # For simplicity, load without compiling, or compile with a default.
                        # loaded_model.compile(optimizer=Adam(), loss='mse') # Example re-compile
                    except Exception as e:
                        logging.error(f"Error loading Keras model {model_name} from {model_file_or_dir}: {e}")

                elif model_file_or_dir.is_file() and model_file_or_dir.name.endswith("_sklearn_model.joblib"):
                    model_name = model_file_or_dir.name.replace("_sklearn_model.joblib", "")
                    try:
                        loaded_model = joblib.load(model_file_or_dir)
                    except Exception as e:
                         logging.error(f"Error loading sklearn model {model_name} from {model_file_or_dir}: {e}")
                
                if model_name and loaded_model is not None:
                    instance.models_[model_name] = loaded_model
        
        # 4. Load feature importances
        fi_path = load_dir / "feature_importances.csv"
        if fi_path.exists():
            instance.feature_importances_ = pd.read_csv(fi_path)

        logging.info("Predictor loaded successfully.")
        return instance

    def _clean_config_for_serialization(self, config_item: Any) -> Any:
        """Recursively cleans config items for JSON serialization (handles Path, sets, classes)."""
        if isinstance(config_item, dict):
            return {k: self._clean_config_for_serialization(v) for k, v in config_item.items()}
        elif isinstance(config_item, list):
            return [self._clean_config_for_serialization(i) for i in config_item]
        elif isinstance(config_item, Path):
            return str(config_item)
        elif isinstance(config_item, type): # Handle class objects (e.g. model_class)
            return f"<class '{config_item.__module__}.{config_item.__name__}'>"
        elif isinstance(config_item, (np.integer, np.floating, np.bool_)): # Numpy scalars
            return config_item.item()
        elif isinstance(config_item, (set)):
            return sorted(list(config_item)) # Convert set to sorted list
        elif callable(config_item) and not isinstance(config_item, tf.keras.layers.Layer): # Non-keras callables
            return f"<function '{config_item.__name__}'>"
        # Basic JSON serializable types
        elif isinstance(config_item, (str, int, float, bool, type(None))):
            return config_item
        else: # For other complex objects, return their string representation
            return str(config_item)

    def export_for_mobile(self, model_name_to_export: Optional[str] = None,
                          output_dir: Optional[Union[str, Path]] = None,
                          tflite_quantization: Optional[str] = 'default') -> Optional[Path]: # 'default', 'float16', 'int8', None
        """Exports a specified Keras model to TFLite format for mobile deployment."""
        if not self.is_fitted_:
            raise NotFittedError("Predictor is not fitted. Cannot export model.")

        if model_name_to_export is None:
            if self.best_model_key_ and isinstance(self.models_.get(self.best_model_key_), tf.keras.Model):
                model_name_to_export = self.best_model_key_
            else: # Find first available Keras model
                for name, model_obj in self.models_.items():
                    if isinstance(model_obj, tf.keras.Model):
                        model_name_to_export = name; break
        
        if not model_name_to_export or model_name_to_export not in self.models_ or \
           not isinstance(self.models_[model_name_to_export], tf.keras.Model):
            logging.error(f"Model '{model_name_to_export}' not found or not a Keras model. Cannot export to TFLite.")
            return None

        keras_model_to_export = self.models_[model_name_to_export]
        
        if output_dir is None:
            export_base_dir = Path(self.config['general']['results_dir']) / self.biomarker_type / "mobile_exports"
            output_dir = export_base_dir / f"{model_name_to_export}_{datetime.now().strftime('%Y%m%d')}"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Exporting model '{model_name_to_export}' to TFLite at {output_dir}")

        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model_to_export)
            
            if tflite_quantization:
                converter.optimizations = [tf.lite.Optimize.DEFAULT] # Latency/size optimization
                if tflite_quantization == 'float16':
                    converter.target_spec.supported_types = [tf.float16]
                elif tflite_quantization == 'int8':
                    # INT8 quantization requires a representative dataset for calibration.
                    # This is a complex step not fully implemented here.
                    # def representative_dataset_gen():
                    #   for _ in range(100): # Example: 100 calibration samples
                    #     # Yield preprocessed data that model expects, e.g., from X_train_fold
                    #     # sample = X_train_fold[np.random.choice(X_train_fold.shape[0])]
                    #     # yield [sample.reshape(1, *sample.shape).astype(np.float32)] # Batch of 1
                    #     pass # Needs implementation
                    # converter.representative_dataset = representative_dataset_gen
                    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    # converter.inference_input_type = tf.int8 # or tf.uint8
                    # converter.inference_output_type = tf.int8 # or tf.uint8
                    logging.warning("INT8 TFLite quantization requires a representative_dataset. Exporting with default optimizations instead.")
            
            tflite_model_content = converter.convert()
            tflite_model_path = output_dir / f"{model_name_to_export}.tflite"
            with open(tflite_model_path, 'wb') as f:
                f.write(tflite_model_content)
            logging.info(f"TFLite model saved to {tflite_model_path}")

            # Save input/output details (optional, but useful for mobile dev)
            interpreter = tf.lite.Interpreter(model_content=tflite_model_content)
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            io_details = {"inputs": input_details, "outputs": output_details}
            with open(output_dir / f"{model_name_to_export}_io_details.json", 'w') as f:
                # Convert numpy types in details to native python types for JSON
                json.dump(self._clean_config_for_serialization(io_details), f, indent=4)

            # Save relevant parts of the preprocessor pipeline (e.g., scaler params)
            # This is complex as the pipeline can have many custom steps not TFLite compatible.
            # Usually, for mobile, preprocessing is reimplemented or simplified.
            # Here, we save scaler params if a standard scaler was used.
            if self.preprocessor_pipeline_ and 'scaler' in self.preprocessor_pipeline_.named_steps:
                scaler_step = self.preprocessor_pipeline_.named_steps['scaler']
                scaler_params = {}
                if hasattr(scaler_step, 'mean_') and scaler_step.mean_ is not None: scaler_params['mean'] = scaler_step.mean_.tolist()
                if hasattr(scaler_step, 'scale_') and scaler_step.scale_ is not None: scaler_params['scale'] = scaler_step.scale_.tolist()
                if hasattr(scaler_step, 'min_') and scaler_step.min_ is not None: scaler_params['min_'] = scaler_step.min_.tolist() # For MinMaxScaler
                if hasattr(scaler_step, 'data_min_') and scaler_step.data_min_ is not None: scaler_params['data_min_'] = scaler_step.data_min_.tolist() # For RobustScaler, MinMaxScaler
                if hasattr(scaler_step, 'data_max_') and scaler_step.data_max_ is not None: scaler_params['data_max_'] = scaler_step.data_max_.tolist()
                if hasattr(scaler_step, 'center_') and scaler_step.center_ is not None: scaler_params['center_'] = scaler_step.center_.tolist() # For RobustScaler
                
                if scaler_params:
                    with open(output_dir / "scaler_params.json", 'w') as f:
                        json.dump(scaler_params, f, indent=4)
            
            # Save feature names expected by the processed model input
            if self.feature_names_processed_:
                 with open(output_dir / "processed_feature_names.json", 'w') as f:
                    json.dump(self.feature_names_processed_, f, indent=4)
            
            return output_dir

        except Exception as e:
            logging.error(f"Error during TFLite conversion for model '{model_name_to_export}': {e}")
            return None

    # Plotting utilities (can be expanded)
    def plot_predictions(self, X_test: Union[pd.DataFrame, np.ndarray], y_test_true: Union[pd.Series, np.ndarray],
                         model_to_use: Optional[str] = None, # 'best', 'ensemble', or specific model_name
                         save_path: Optional[Union[str, Path]] = None, show_plot: bool = True):
        """Plots true vs. predicted values."""
        if not self.is_fitted_: raise NotFittedError("Fit predictor first.")

        use_best = (model_to_use == 'best') or (model_to_use is None and self.config['model_selection']['architecture_type'] != 'ensemble')
        
        y_pred, y_uncertainty = self.predict(X_test, return_uncertainty=True, use_best_single_model=use_best)
        y_true_np = y_test_true.to_numpy() if isinstance(y_test_true, pd.Series) else np.asarray(y_test_true)

        plt.figure(figsize=(10, 8))
        plt.scatter(y_true_np, y_pred, alpha=0.6, label="Predictions")
        
        min_val = min(np.min(y_true_np), np.min(y_pred))
        max_val = max(np.max(y_true_np), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal")

        if y_uncertainty is not None and not np.all(np.isnan(y_uncertainty)):
            # Sort by true values for cleaner error bar plot if desired, or just plot as is
            sorted_indices = np.argsort(y_true_np) # Optional: for cleaner plot
            plt.fill_between(y_true_np[sorted_indices],
                             (y_pred - 1.96 * y_uncertainty)[sorted_indices],
                             (y_pred + 1.96 * y_uncertainty)[sorted_indices],
                             color='gray', alpha=0.3, label="95% CI (Uncertainty)")

        metrics_eval = self.evaluate(X_test, y_test_true, use_best_single_model=use_best)
        metrics_text = "\n".join([f"{k.upper()}: {v:.3f}" for k, v in metrics_eval.items() if k in ['mae', 'rmse', 'r2']])
        
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.xlabel(f"True {self.biomarker_type.replace('_',' ').title()}")
        plt.ylabel(f"Predicted {self.biomarker_type.replace('_',' ').title()}")
        plt.title(f"{self.biomarker_type.replace('_',' ').title()} Prediction Scatter Plot")
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            logging.info(f"Prediction plot saved to {save_path}")
        if show_plot: plt.show()
        plt.close()

    def plot_feature_importances(self, top_n: int = 20,
                                 save_path: Optional[Union[str, Path]] = None, show_plot: bool = True):
        """Plots aggregated feature importances."""
        if not self.is_fitted_ or self.feature_importances_ is None or self.feature_importances_.empty:
            logging.warning("Feature importances not available or predictor not fitted.")
            return

        fi_to_plot = self.feature_importances_.head(top_n)
        plt.figure(figsize=(10, max(6, top_n * 0.3))) # Adjust height based on num features
        plt.barh(fi_to_plot['feature'], fi_to_plot['importance'], color='skyblue')
        plt.xlabel("Mean Importance Score")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} Feature Importances for {self.biomarker_type.replace('_',' ').title()}")
        plt.gca().invert_yaxis() # Highest importance at the top
        plt.grid(True, axis='x', linestyle=':')
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)
            logging.info(f"Feature importance plot saved to {save_path}")
        if show_plot: plt.show()
        plt.close()


# =============== MAIN EXECUTION EXAMPLE (Illustrative) ===============
if __name__ == '__main__':
    logging.info("Starting Physics-Informed Optical Biomarker Prediction Framework Demo")

    # --- 1. Generate Synthetic Data ---
    # More realistic synthetic data: simulate multiple PPG channels and features
    n_samples = 500
    n_raw_features_per_channel = 1000 # E.g., 10 seconds at 100Hz
    
    # Simulate raw signal columns (e.g., red_raw, ir_raw)
    X_synth_df = pd.DataFrame()
    channel_names = ['raw_red_channel', 'raw_ir_channel', 'raw_green_channel']
    for ch_name in channel_names:
        # Each "signal" is a list of arrays for this demo structure for PPGSignalPreprocessor
        signals_list = []
        for _ in range(n_samples):
            base_signal = np.sin(np.linspace(0, 20*np.pi, n_raw_features_per_channel)) # Base oscillation
            noise = np.random.normal(0, 0.2, n_raw_features_per_channel)
            drift = np.linspace(0, np.random.rand()*2-1, n_raw_features_per_channel) # Random drift
            signals_list.append(base_signal + noise + drift + np.random.rand()*5) # Add some offset
        X_synth_df[ch_name] = signals_list

    # Add some dummy metadata features that will be passed through
    X_synth_df['age'] = np.random.randint(20, 70, n_samples)
    X_synth_df['gender_male'] = np.random.randint(0, 2, n_samples)

    # Simulate a target biomarker (e.g., glucose) correlated with some aspect of the signals
    # For simplicity, let's make it correlated with the mean amplitude of 'raw_ir_channel'
    y_synth = pd.Series([np.mean(np.abs(s - np.mean(s))) for s in X_synth_df['raw_ir_channel']], name='glucose')
    y_synth = y_synth * 50 + 70 + np.random.normal(0, 15, n_samples) # Scale to glucose-like range + noise
    y_synth = np.clip(y_synth, 40, 400)
    
    logging.info(f"Generated synthetic data: X shape {X_synth_df.shape}, y mean {y_synth.mean():.2f}")

    # --- 2. Configure and Initialize Predictor ---
    # Custom config for the demo
    demo_config = {
        "general": {"verbose": 1, "results_dir": "demo_predictor_results"},
        "data_preprocessing": {
            "steps": [
                {"name": "signal_preprocessor", "active": True, "params": {
                    "signal_column_prefixes": ['raw_red', 'raw_ir', 'raw_green'], # Match synthetic data
                    "sampling_rate_hz": 100.0,
                    "segment_length_s": 5.0, # Shorter segments for demo speed
                    "keep_original_columns": True # To keep 'age', 'gender_male'
                }},
                {"name": "wavelet_features", "active": True, "params": {"max_level": 3}}, # Fewer levels for speed
                {"name": "spectral_features", "active": True},
                {"name": "multispectral_features", "active": True, "params": {
                    # These suffixes should match outputs of wavelet/spectral if they are inputs to MSF
                    # For demo, assume basic stats are used or MSF adapts
                    "wavelength_channel_prefixes": ['proc_raw_red', 'proc_raw_ir', 'proc_raw_green'], # From signal_preprocessor
                    "mean_suffix": '_wavelet_app_L3_mean', # Example
                    "std_suffix": '_wavelet_app_L3_std',   # Example
                }},
                {"name": "imputer", "active": True, "params": {"strategy": "median"}},
                {"name": "scaler", "active": True, "params": {"type": "robust"}}
            ]
        },
        "model_selection": {
            "architecture_type": "single_dl", # Faster for demo
            "primary_model_name": "physics_informed_cnn", # Use the PI model
            "dl_models": {
                "physics_informed_cnn": {"active": True, "builder": "build_physics_informed_model", "config": {
                    "base_model_type": "cnn",
                    "base_model_config": { # Simpler CNN backbone for demo
                        "num_filters_list": [16, 32], "kernel_sizes_list": [5, 3], "res_strides_list": [1,1],
                        "res_activation": 'relu', "res_block_dropout": 0.1, "use_squeeze_excitation": False,
                        "attention_heads": 0, "global_pooling_type": 'avg',
                        "final_dense_units": [16], "dropout_final_dense": 0.1, "dense_activation": 'relu',
                        "l2_reg_cnn": 1e-5, "l2_reg_dense": 1e-4
                    },
                    "physics_paths": [ # Example for Glucose
                        {"type": "scattering", "params": {
                            "biomarker_ri_effects": {'glucose_mg_dl': OPTICAL_CONSTANTS['glucose']['refractive_index_change_per_mg_dl']},
                         }, "feature_map_dense_units": [8]}
                    ],
                    "combination_method": 'concatenate_dense', "final_dense_units_combiner": [16],
                    "l2_reg": 1e-5, "output_activation_type": 'scaled_sigmoid'
                }}
            },
             "ml_models": {"random_forest": {"active": False}, "gradient_boosting": {"active": False}} # Turn off ML for speed
        },
        "training": {
            "epochs": 10, "batch_size": 32, # Short training for demo
            "callbacks": {
                "early_stopping": {"patience": 5},
                "reduce_lr_on_plateau": {"patience": 3}
            }
        },
        "hyperparameter_optimization": {"active": False} # Turn off HPO for demo speed
    }

    predictor = PhysicsInformedBiomarkerPredictor(biomarker_type='glucose', config=demo_config)

    # --- 3. Split Data and Fit Predictor ---
    X_train, X_test, y_train, y_test = train_test_split(X_synth_df, y_synth, test_size=0.25, random_state=predictor.config['general']['random_state'])
    
    logging.info("Fitting the predictor...")
    predictor.fit(X_train, y_train, X_val=X_test, y_val=y_test) # Use test as val for demo simplicity

    # --- 4. Evaluate Predictor ---
    logging.info("Evaluating the predictor on the test set...")
    test_metrics = predictor.evaluate(X_test, y_test)
    logging.info(f"Test set evaluation metrics: {test_metrics}")

    # --- 5. Make Predictions and Plot ---
    if predictor.is_fitted_:
        y_pred_test, y_uncertainty_test = predictor.predict(X_test, return_uncertainty=True)
        
        results_plot_dir = Path(predictor.config['general']['results_dir']) / predictor.biomarker_type / "plots"
        predictor.plot_predictions(X_test, y_test,
                                   save_path=results_plot_dir / "demo_predictions_scatter.png",
                                   show_plot=False) # Set show_plot=True for interactive display
        
        if predictor.feature_importances_ is not None:
            predictor.plot_feature_importances(top_n=15,
                                               save_path=results_plot_dir / "demo_feature_importances.png",
                                               show_plot=False)

    # --- 6. Save and Load Predictor ---
    if predictor.is_fitted_:
        logging.info("Saving the predictor...")
        saved_path = predictor.save()
        logging.info(f"Predictor saved to: {saved_path}")

        logging.info("Loading the predictor back...")
        try:
            loaded_predictor = PhysicsInformedBiomarkerPredictor.load(saved_path)
            logging.info("Predictor loaded successfully. Making a test prediction.")
            loaded_pred_test = loaded_predictor.predict(X_test.iloc[:5]) # Predict on a few samples
            logging.info(f"Loaded predictor predictions (first 5): {loaded_pred_test}")
        except Exception as e:
            logging.error(f"Error loading or testing loaded predictor: {e}", exc_info=True)


    # --- 7. Export for Mobile (if a Keras model was trained) ---
    if predictor.is_fitted_ and any(isinstance(m, tf.keras.Model) for m in predictor.models_.values()):
        logging.info("Exporting model for mobile...")
        # Export best_model_key if it's a Keras model, or the primary_model_name if configured.
        model_to_export_name = None
        if predictor.best_model_key_ and isinstance(predictor.models_.get(predictor.best_model_key_), tf.keras.Model):
            model_to_export_name = predictor.best_model_key_
        elif predictor.config['model_selection']['primary_model_name'] in predictor.models_ and \
             isinstance(predictor.models_[predictor.config['model_selection']['primary_model_name']], tf.keras.Model):
            model_to_export_name = predictor.config['model_selection']['primary_model_name']

        if model_to_export_name:
            mobile_export_path = predictor.export_for_mobile(model_name_to_export=model_to_export_name)
            if mobile_export_path:
                logging.info(f"Model exported for mobile to: {mobile_export_path}")
        else:
            logging.warning("No suitable Keras model found to export for mobile.")

    logging.info("Framework Demo Finished.")