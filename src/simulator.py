# src/simulator.py

from __future__ import annotations
from typing import Optional, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import KernelConfig
from .data_io import Characterization, SimpleChar, FullCharDataType
from .kernel import (
    calculate_kernel_current_scaling_factor,
    get_vtg_for_target_iph,
    get_iph_for_vtg_interpolated,
)
import logging

log = logging.getLogger(__name__)

# --- Data Structures for Results ---

@dataclass(frozen=True)
class FixedPowerResult:
    """Holds the results of a fixed-power calculation."""
    voltages: np.ndarray
    currents: np.ndarray

@dataclass(frozen=True)
class DynamicModeResult:
    """Holds the results of the dynamic regime optimization."""
    voltages: np.ndarray
    loss: float
    ref_nd: float

# --- The Simulator Class ---

class Simulator:
    """
    Handles all the core scientific computations for the kernel simulation,
    decoupled from the GUI.
    """
    def __init__(self, full_char_data: FullCharDataType, config: KernelConfig):
        if not full_char_data:
            raise ValueError("Simulator cannot be initialized with empty characterization data.")
        self.full_char_data = full_char_data
        self.cfg = config
        self.min_x = min(k[0] for k in self.full_char_data.keys())
        self.min_y = min(k[1] for k in self.full_char_data.keys())
        self.nd_min_map = 2.0
        self.nd_max_map = 5.0        
        self.power_map = self._create_perceptual_power_map()

    def run_convolution(self, normalized_input: np.ndarray, kernel_weights: np.ndarray) -> np.ndarray:
        """
        Performs a standard convolution and returns the raw output in Amps.
        """
        H, W = normalized_input.shape
        kH, kW = kernel_weights.shape
        pad_h, pad_w = kH // 2, kW // 2

        padded = np.pad(normalized_input, ((pad_h, pad_h), (pad_w, pad_w)), mode="symmetric")
        raw_out = np.zeros((H, W), dtype=np.float64)

        for r in range(H):
            for c in range(W):
                patch = padded[r : r + kH, c : c + kW]
                raw_out[r, c] = float(np.sum(patch * kernel_weights))
        
        return raw_out

    def run_fixed_power_calculation(
        self, unitless_kernel: np.ndarray, power_level: float, nd5_voltage_cache: Optional[dict], preset_name: str
    ) -> FixedPowerResult:
        """
        Calculates the required voltages and resulting currents for a given kernel
        at a fixed optical power.
        """
        final_voltages = np.zeros((3, 3))

        # Use pre-computed cache if available for ND=5
        if power_level == 5.0 and nd5_voltage_cache and preset_name in nd5_voltage_cache:
            final_voltages = np.array(nd5_voltage_cache[preset_name])
        else:
            # Standard calculation
            simple_chars = {
                pc: SimpleChar(i_max_pos=cd.i_ph_vs_nd_original[power_level].max(), i_max_neg=cd.i_ph_vs_nd_original[power_level].min())
                for pc, cd in self.full_char_data.items() if power_level in cd.nd_axis
            }
            if not simple_chars:
                raise ValueError(f"No characterization data available for ND={power_level}")

            s_factor = calculate_kernel_current_scaling_factor(unitless_kernel, simple_chars)
            target_currents = unitless_kernel * s_factor

            for r_k, c_k in np.ndindex(3, 3):
                pixel_coord = (self.min_x + c_k, self.min_y + r_k)
                char = self.full_char_data.get(pixel_coord)
                target_iph = target_currents[r_k, c_k]
                if char:
                    final_voltages[r_k, c_k] = get_vtg_for_target_iph(target_iph, char, power_level)

        # Calculate the actual currents produced by these voltages
        actual_currents = np.zeros((3, 3))
        for r_k, c_k in np.ndindex(3, 3):
            pixel_coord = (self.min_x + c_k, self.min_y + r_k)
            char = self.full_char_data.get(pixel_coord)
            vtg = final_voltages[r_k, c_k]
            if char and power_level in char.nd_axis:
                actual_currents[r_k, c_k] = get_iph_for_vtg_interpolated(vtg, power_level, char)

        return FixedPowerResult(voltages=final_voltages, currents=actual_currents)

    def run_dynamic_mode_optimization(
        self, unitless_kernel: np.ndarray, progress_callback: Optional[Callable[[str], None]] = None
    ) -> DynamicModeResult:
        """
        Finds a single, fixed set of gate voltages that best preserves the kernel's
        shape across a range of optical powers.
        """
        if progress_callback is None:
            progress_callback = lambda msg: None  # No-op if no callback is provided

        max_abs_weight = np.max(np.abs(unitless_kernel))
        if max_abs_weight < 1e-9:
            raise ValueError("Kernel weights are all zero.")
        ideal_unitless_shape = unitless_kernel / max_abs_weight

        all_results = []
        all_measured_vtgs = pd.concat([char.i_ph_vs_nd_original.index.to_series() for char in self.full_char_data.values()]).unique()
        sorted_unique_vtgs = np.sort(all_measured_vtgs)

        for ref_power_for_search in self.cfg.REFERENCE_ND_LEVELS_TO_TEST:
            progress_callback(f"Optimizing with ND={ref_power_for_search} as reference...")

            for vtg_candidate in sorted_unique_vtgs:
                s_factors = []
                for r_k, c_k in np.ndindex(3, 3):
                    pixel_coord = (np.int64(self.min_x + c_k), np.int64(self.min_y + r_k))
                    char = self.full_char_data.get(pixel_coord)
                    weight = unitless_kernel[r_k, c_k]
                    if not char or abs(weight) < 1e-9: continue
                    
                    iph_at_vtg = get_iph_for_vtg_interpolated(vtg_candidate, ref_power_for_search, char)
                    if (weight > 0 and iph_at_vtg > 1e-12): s_factors.append(iph_at_vtg / weight)
                    elif (weight < 0 and iph_at_vtg < -1e-12): s_factors.append(abs(iph_at_vtg) / abs(weight))

                if not s_factors: continue
                s_achievable = min(s_factors)
                target_currents = unitless_kernel * s_achievable
                
                vtg_candidate_set = np.zeros((3, 3))
                for r_k, c_k in np.ndindex(3, 3):
                    pixel_coord = (np.int64(self.min_x + c_k), np.int64(self.min_y + r_k))
                    char = self.full_char_data.get(pixel_coord)
                    if char: vtg_candidate_set[r_k, c_k] = get_vtg_for_target_iph(target_currents[r_k, c_k], char, ref_power_for_search)

                total_shape_error = 0
                powers_to_test = [2.0, 3.0, 4.0]
                for power in powers_to_test:
                    achieved_currents = self.calculate_currents_for_voltages(vtg_candidate_set, power)
                    max_abs_achieved = np.max(np.abs(achieved_currents))
                    if max_abs_achieved < 1e-12: continue
                    achieved_unitless_shape = achieved_currents / max_abs_achieved
                    total_shape_error += np.sum((ideal_unitless_shape - achieved_unitless_shape)**2)
                
                all_results.append({'total_loss': total_shape_error, 'voltages': vtg_candidate_set, 'ref_nd': ref_power_for_search})

        if not all_results:
            raise RuntimeError("Optimization failed. Could not find a stable voltage set.")

        best_result = min(all_results, key=lambda x: x['total_loss'])
        return DynamicModeResult(
            voltages=best_result['voltages'],
            loss=best_result['total_loss'],
            ref_nd=best_result['ref_nd']
        )
    
    def _create_perceptual_power_map(self):
        """Creates a mapping from normalized intensity [0,1] to an effective ND filter value."""
        gamma = 2.2 if self.cfg.dynamic_mode_use_gamma_correction else 1.0
        transmission_max = 10**(-self.nd_min_map)
        transmission_min = 10**(-self.nd_max_map)
        
        def perceptual_map(perceptual_intensity):
            linear_light = perceptual_intensity**gamma
            transmission = transmission_min + (transmission_max - transmission_min) * linear_light
            return -np.log10(transmission + 1e-10)

        log.info(f"Created perceptually correct power map for ND range [{self.nd_min_map}, {self.nd_max_map}]")
        return perceptual_map

    def get_adaptive_kernel_weights(self, stimulus_patch_0_1: np.ndarray, fixed_voltages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the adaptive kernel weights (currents) based on
        the input image patch intensity and the fixed optimal gate voltages.
        """
        adaptive_weights = np.zeros((3, 3))
        effective_nds = np.zeros((3, 3))

        for r_k, c_k in np.ndindex(3, 3):
            vtg = fixed_voltages[r_k, c_k]
            intensity = stimulus_patch_0_1[r_k, c_k]
            ideal_nd = self.power_map(intensity)
            effective_nds[r_k, c_k] = ideal_nd

            pixel_coord = (np.int64(self.min_x + c_k), np.int64(self.min_y + r_k))
            char = self.full_char_data.get(pixel_coord)
            if char:
                adaptive_weights[r_k, c_k] = get_iph_for_vtg_interpolated(vtg, ideal_nd, char)

        return adaptive_weights, effective_nds
    
    def calculate_currents_for_voltages(self, voltages: np.ndarray, power_level: float) -> np.ndarray:
        """Helper to calculate currents for a given voltage matrix and power level."""
        currents = np.zeros((3, 3))
        for r_k, c_k in np.ndindex(3, 3):
            pixel_coord = (self.min_x + c_k, self.min_y + r_k)
            char = self.full_char_data.get(pixel_coord)
            if char:
                currents[r_k, c_k] = get_iph_for_vtg_interpolated(voltages[r_k, c_k], power_level, char)
        return currents