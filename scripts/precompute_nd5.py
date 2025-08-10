# precompute_nd5.py

import pandas as pd
import numpy as np
import os
import sys
import json
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to the Python path to allow importing from 'src'
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.config import KernelConfig
from src.data_io import load_all_characterization_from_folder, SimpleChar
from src.presets import ALL_PRESET_KERNELS 
from src.kernel import calculate_kernel_current_scaling_factor
from src.fitting_models import (
    gaussian_with_baseline, 
    lorentzian_with_baseline,
    double_gaussian_with_baseline,
    double_lorentzian_with_baseline,
    piecewise_gaussian_with_baseline,
    calculate_r_squared
)

print("--- ND=5 Voltage Pre-computation Script ---")

# --- Configuration ---
cfg = KernelConfig()
CLEAN_ND_FOR_PRIOR = 4
NOISY_ND_TO_FIT = 5
OUTPUT_FILENAME = "nd5_voltage_cache.json"

# --- Helper function to find all intersections ---
def find_intersections(x_data, y_data, target_y):
    intersections = []
    for i in range(len(x_data) - 1):
        x1, y1, x2, y2 = x_data[i], y_data[i], x_data[i+1], y_data[i+1]
        if (y1 - target_y) * (y2 - target_y) < 0 and y2 - y1 != 0:
            x_crossing = x1 + (x2 - x1) * (target_y - y1) / (y2 - y1)
            intersections.append(x_crossing)
    return intersections

# --- Main Logic ---
full_char_data = load_all_characterization_from_folder(cfg)
if not full_char_data:
    print("FATAL: Could not load characterization data. Exiting.")
    sys.exit(1)

clean_char_data = {pc: cd for pc, cd in full_char_data.items() if CLEAN_ND_FOR_PRIOR in cd.nd_axis}
noisy_char_data = {pc: cd for pc, cd in full_char_data.items() if NOISY_ND_TO_FIT in cd.nd_axis}

all_pixel_coords = sorted(noisy_char_data.keys(), key=lambda p: (p[1], p[0]))
final_voltage_cache = {}

# --- Step 1: Find the Best Fit for Each Pixel ---
print("\n--- Step 1: Finding the best predictive model for each pixel ---")
pixel_fits = {}
# ... (This entire fitting block is unchanged and correct) ...
for pixel_coord in all_pixel_coords:
    print(f"  - Analyzing Pixel {pixel_coord}...")
    clean_data = clean_char_data.get(pixel_coord).i_ph_vs_nd_original[CLEAN_ND_FOR_PRIOR] if pixel_coord in clean_char_data else pd.Series()
    noisy_data = noisy_char_data[pixel_coord].i_ph_vs_nd_original[NOISY_ND_TO_FIT]
    priors = {}
    if not clean_data.empty and len(clean_data) > 8:
        models_for_priors = {"S-Gauss": gaussian_with_baseline, "D-Gauss": double_gaussian_with_baseline, "D-Lorentz": double_lorentzian_with_baseline}
        for name, func in models_for_priors.items():
            try:
                p0 = [0,0,clean_data.min(),clean_data.idxmin(),1.0];
                if name.startswith("D-"): p0 = [0,0,clean_data.min()*0.7,clean_data.idxmin()-0.5,1.0, clean_data.min()*0.3,clean_data.idxmin()+0.5,1.0]
                popt, _ = curve_fit(func, clean_data.index, clean_data.values, p0=p0); priors[name] = popt
            except RuntimeError: pass
    if not priors: priors['Rough'] = [0,0,noisy_data.min(), noisy_data.idxmin(), 1.0, noisy_data.min()*0.1, noisy_data.idxmin()+1, 1.0]
    all_fit_results = []
    final_models_to_test = {"S-Gauss": gaussian_with_baseline, "S-Lorentz": lorentzian_with_baseline, "D-Gauss": double_gaussian_with_baseline, "D-Lorentz": double_lorentzian_with_baseline, "P-Gauss": piecewise_gaussian_with_baseline}
    for prior_name, prior_popt in priors.items():
        center_prior = prior_popt[3]; width_prior = abs(prior_popt[4])
        for model_name, model_func in final_models_to_test.items():
            try:
                amp_guess = noisy_data.min() - np.median(noisy_data); p0 = []
                if model_name.startswith("S-"): p0 = [0, 0, amp_guess, center_prior, width_prior]
                elif model_name.startswith("D-"): p0 = [0, 0, amp_guess*0.7, center_prior-width_prior/2, width_prior, amp_guess*0.3, center_prior+width_prior/2, width_prior]
                elif model_name.startswith("P-"): p0 = [0, 0, np.median(noisy_data), amp_guess, center_prior, width_prior]
                popt_final, _ = curve_fit(model_func, noisy_data.index, noisy_data.values, p0=p0, maxfev=10000)
                r2 = calculate_r_squared(noisy_data.values, model_func(noisy_data.index, *popt_final))
                all_fit_results.append({'r2': r2, 'name': model_name, 'prior_name': prior_name, 'popt': popt_final, 'func': model_func})
            except (RuntimeError, ValueError): pass
    if all_fit_results:
        all_fit_results.sort(key=lambda item: item['r2'], reverse=True)
        pixel_fits[pixel_coord] = all_fit_results[0]
        print(f"    -> Best fit is '{pixel_fits[pixel_coord]['name']}' with R²={pixel_fits[pixel_coord]['r2']:.4f}")

# --- Step 2: Calculate Voltage Maps for each Kernel ---
print("\n--- Step 2: Calculating voltage maps for each preset kernel ---")

# --- Create a directory for debug plots ---
debug_plot_dir = "precompute_debug_plots"
os.makedirs(debug_plot_dir, exist_ok=True)
print(f"  - Saving debug plots to '{debug_plot_dir}/'")
min_x, min_y = min(k[0] for k in full_char_data.keys()), min(k[1] for k in full_char_data.keys())

robust_simple_chars = {}
for pixel_coord, fit_info in pixel_fits.items():
    char = noisy_char_data[pixel_coord]
    x_fit = np.linspace(char.v_tg.min(), char.v_tg.max(), 1000)
    y_fit = fit_info['func'](x_fit, *fit_info['popt'])
    robust_simple_chars[pixel_coord] = SimpleChar(i_max_pos=np.max(y_fit), i_max_neg=np.min(y_fit))

for kernel_name, unitless_kernel in ALL_PRESET_KERNELS.items():
    print(f"  - Pre-calculating for kernel: '{kernel_name}'")
    s_factor = calculate_kernel_current_scaling_factor(unitless_kernel, robust_simple_chars)
    target_currents = unitless_kernel * s_factor
    print(f"    -> Calculated S-Factor: {s_factor:.3e}")
    
    voltage_matrix = np.zeros((3, 3))
    
    fig_debug, axes_debug = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
    fig_debug.suptitle(f"Voltage Calculation for '{kernel_name}' (ND=5)", fontsize=16)

    for r_k, c_k in np.ndindex(3, 3):
        ax_debug = axes_debug[r_k, c_k]
        pixel_coord = (min_x + c_k, min_y + r_k)
        target_iph = target_currents[r_k, c_k]
        weight = unitless_kernel[r_k, c_k]
        
        ax_debug.set_title(f"Pixel {pixel_coord}")
        fit_info = pixel_fits.get(pixel_coord)
        char = noisy_char_data.get(pixel_coord)

        if char and fit_info:
            noisy_data = char.i_ph_vs_nd_original[NOISY_ND_TO_FIT]
            ax_debug.plot(noisy_data.index, noisy_data.values, 'ko', markersize=2, alpha=0.4, label='Raw Data')
            x_fit = np.linspace(noisy_data.index.min(), noisy_data.index.max(), 500)
            y_fit = fit_info['func'](x_fit, *fit_info['popt'])
            ax_debug.plot(x_fit, y_fit, 'r-', label=f"Best Fit ({fit_info['name']})")
            
            final_vtg = 0.0
            # --- *** THIS IS THE DEFINITIVE CORRECTED LOGIC *** ---
            crossings = find_intersections(x_fit, y_fit, target_iph)
            
            if crossings:
                # If the weight is zero, the target is zero. Choose the zero-crossing closest to 0V.
                if abs(weight) < 1e-9:
                    final_vtg = min(crossings, key=lambda c: abs(c - 0.0))
                    print(f"    - Pixel {pixel_coord}: Weight is zero. Chose zero-crossing at: {final_vtg:.2f} V")
                elif weight < 0: # Negative weight, find dip, choose leftmost crossing
                    final_vtg = min(crossings)
                    print(f"    - Pixel {pixel_coord}: Found crossings at {np.round(crossings, 2)}. Chose leftmost: {final_vtg:.2f} V")
                else: # Positive weight, find peak, choose crossing closest to the peak
                    peak_vtg = x_fit[np.argmax(y_fit)]
                    final_vtg = min(crossings, key=lambda c: abs(c - peak_vtg))
                    print(f"    - Pixel {pixel_coord}: Found crossings at {np.round(crossings, 2)}. Chose closest to peak: {final_vtg:.2f} V")
            else:
                # BEST EFFORT FALLBACK: Saturate at the curve's extremum
                if target_iph > np.max(y_fit):
                    final_vtg = x_fit[np.argmax(y_fit)]
                    print(f"    -> WARNING: No intersection for pixel {pixel_coord}. Using Vtg at MAX peak: {final_vtg:.2f} V")
                else: 
                    final_vtg = x_fit[np.argmin(y_fit)]
                    print(f"    -> WARNING: No intersection for pixel {pixel_coord}. Using Vtg at MIN dip: {final_vtg:.2f} V")
            
            voltage_matrix[r_k, c_k] = final_vtg
            # --- END OF FIX ---

            ax_debug.axhline(target_iph, color='blue', linestyle='--', label=f'Target Iph: {target_iph:.2e} A')
            ax_debug.axvline(final_vtg, color='green', linestyle=':', label=f'Found Vtg: {final_vtg:.2f} V')
            ax_debug.legend()
        else:
            ax_debug.text(0.5, 0.5, "No Data / Fit Failed", ha='center', va='center', transform=ax_debug.transAxes)

    final_voltage_cache[kernel_name] = voltage_matrix.tolist()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # Save the figure 
    fig_filename = os.path.join(debug_plot_dir, f"debug_{kernel_name}.png")
    plt.savefig(fig_filename, dpi=150)
    plt.close(fig_debug) # Close the figure to free up memory

# --- Step 3: Save to JSON file ---
with open(OUTPUT_FILENAME, 'w') as f:
    json.dump(final_voltage_cache, f, indent=4)

print(f"\nSUCCESS: Pre-computation complete. Results for {len(final_voltage_cache)} kernels saved to '{OUTPUT_FILENAME}'.")