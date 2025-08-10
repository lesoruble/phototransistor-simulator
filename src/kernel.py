# src/kernel.py

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from .data_io import Characterization, SimpleChar
import logging

log = logging.getLogger(__name__)

def calculate_kernel_current_scaling_factor(
    
    unitless_kernel: np.ndarray,
    pixel_simple_chars: Dict[Tuple[int, int], SimpleChar]
) -> float:
    """
    Calculates the scaling factor 'S' to convert a unitless kernel into a physical
    current-based kernel. 'S' is chosen to be the minimum possible value that ensures
    no target current exceeds the maximum achievable current for any pixel.
    
    S = min( I_max_achievable / |weight| ) for all non-zero weights.
    """
    if not pixel_simple_chars or np.all(unitless_kernel == 0): return 0.0
    
    possible_s_factors = []
    min_x = min(k[0] for k in pixel_simple_chars.keys())
    min_y = min(k[1] for k in pixel_simple_chars.keys())

    for r_k, c_k in np.ndindex(unitless_kernel.shape):
        weight = unitless_kernel[r_k, c_k]
        pixel_coord = (min_x + c_k, min_y + r_k)
        
        if pixel_coord not in pixel_simple_chars: continue
        
        char = pixel_simple_chars[pixel_coord]
        
        if weight > 1e-9:
            if char.i_max_pos > 1e-12:
                possible_s_factors.append(char.i_max_pos / weight)
        elif weight < -1e-9:
            if abs(char.i_max_neg) > 1e-12:
                possible_s_factors.append(abs(char.i_max_neg / weight))
    
    if not possible_s_factors:
        log.warning("No valid S-Factors could be calculated. Kernel may be unachievable.")
        return 0.0
        
    s_factor = np.min(possible_s_factors)
    log.info(f"Possible S-Factors: {np.array2string(np.array(possible_s_factors), formatter={'float_kind':lambda x: '%.2e' % x})}")
    log.info(f"Chosen MINIMUM S-Factor: {s_factor:.3e}")
    
    return s_factor

def get_vtg_for_target_iph(target_iph: float, char: Characterization, nd_level: float) -> float:
    """
    Finds the gate voltage (V_tg) required to produce a specific target photocurrent
    at a given optical power (nd_level), using the original-spacing data.

    This function robustly handles non-monotonic I-V curves by finding all points
    where the curve crosses the target current and selecting the first crossing.
    """
    if char is None or nd_level not in char.i_ph_vs_nd_original.columns:
        return 0.0
    
    iph_curve = char.i_ph_vs_nd_original[nd_level]
    voltages = iph_curve.index.to_numpy()
    currents = iph_curve.to_numpy()

    # Create a shifted curve to find zero crossings
    shifted_currents = currents - target_iph

    # Find indices where the sign of the shifted curve changes (i.e., crosses zero)
    crossings = np.where(np.diff(np.sign(shifted_currents)))[0]

    if crossings.size == 0:
        # No exact crossing found. Find the voltage corresponding to the closest current.
        closest_idx = np.argmin(np.abs(shifted_currents))
        return voltages[closest_idx]

    # Of all crossings, find the best one. Often the first is desired.
    # We will interpolate for higher precision at the first crossing point.
    best_crossing_idx = crossings[0]
    
    # Linear interpolation for V_tg at the crossing point
    # y = mx + c => x = (y - c) / m
    y1, y2 = shifted_currents[best_crossing_idx], shifted_currents[best_crossing_idx + 1]
    x1, x2 = voltages[best_crossing_idx], voltages[best_crossing_idx + 1]

    # We want to find x where y=0
    # Using the two-point form of a line: (y - y1) / (x - x1) = (y2 - y1) / (x2 - x1)
    # Set y=0 and solve for x: x = x1 - y1 * (x2 - x1) / (y2 - y1)
    if abs(y2 - y1) < 1e-15: # Avoid division by zero if the line is flat
        return x1 
        
    vtg = x1 - y1 * (x2 - x1) / (y2 - y1)
    return float(vtg)


def get_iph_for_vtg_interpolated(vtg: float, nd_level: float, char: Characterization) -> float:
    """
    Performs 2D interpolation on the characterization data. It finds the photocurrent
    for a given V_tg and a (potentially non-measured) continuous ND_level.
    
    This works by:
    1. Finding the two measured ND levels that bound the target `nd_level`.
    2. For each of these two ND levels, interpolating along the V_tg axis to find the current.
    3. Performing a final linear interpolation between these two currents to get the final value.
    """

    if char is None: return 0.0
    
    iv_data = char.i_ph_vs_nd_original
    available_nds = iv_data.columns.to_numpy(dtype=float)

    # Handle extrapolation (clamping to the nearest available ND curve)
    if nd_level <= available_nds[0]:
        nd_to_use = available_nds[0]
        curve = iv_data[nd_to_use]
        return np.interp(vtg, curve.index, curve.values, left=curve.values[0], right=curve.values[-1])
    
    if nd_level >= available_nds[-1]:
        nd_to_use = available_nds[-1]
        curve = iv_data[nd_to_use]
        return np.interp(vtg, curve.index, curve.values, left=curve.values[0], right=curve.values[-1])

    # Find the bounding ND levels for interpolation
    idx_high = np.searchsorted(available_nds, nd_level)
    idx_low = idx_high - 1
    nd_low = available_nds[idx_low]
    nd_high = available_nds[idx_high]

    # Interpolate along V_tg for the lower ND curve
    curve_low = iv_data[nd_low]
    i_ph_low = np.interp(vtg, curve_low.index, curve_low.values, left=curve_low.values[0], right=curve_low.values[-1])

    # Interpolate along V_tg for the upper ND curve
    curve_high = iv_data[nd_high]
    i_ph_high = np.interp(vtg, curve_high.index, curve_high.values, left=curve_high.values[0], right=curve_high.values[-1])
    
    # Final interpolation along the ND axis
    final_i_ph = np.interp(nd_level, [nd_low, nd_high], [i_ph_low, i_ph_high])
    return final_i_ph
