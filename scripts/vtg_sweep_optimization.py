# plotter_dynamic_regime_v2.py

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import json

# --- Add project root to path to allow src imports ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(project_root))

from src.config import KernelConfig
from src.data_io import load_all_characterization_from_folder
from src.kernel import get_vtg_for_target_iph, get_iph_for_vtg_interpolated
from src.presets import ALL_PRESET_KERNELS

print("--- Final Targeted Dynamic Regime Optimization with ND=5 Cache ---")

# --- Configuration ---
cfg = KernelConfig()
KERNEL_TO_OPTIMIZE = "MEXICAN_HAT" 
REFERENCE_ND_LEVELS_TO_TEST = [2.0, 3.0, 4.0]
powers_to_test = [2.0, 3.0, 4.0, 5.0]

output_dir = os.path.join(project_root, "optimization_plots")
os.makedirs(output_dir, exist_ok=True)
print(f"Saving plots to: {output_dir}")

# --- Font Size Configuration for Plots ---
FONT_SIZES = {
    'title': 22,
    'label': 18,
    'legend': 16,
    'ticks': 16,
    'legend_title': 16,
    'kernel_title': 16,
    'kernel_text': 12
}

# --- Helper function for plotting kernels ---
def plot_kernel_on_ax(ax, data, title, vmin, vmax):
    """
    Plots a 3x3 kernel on a given matplotlib axis with text annotations.
    """
    im = ax.imshow(data, cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_title(title, fontsize=FONT_SIZES['kernel_title'])
    
    v_range = vmax - vmin
    for r in range(3):
        for c in range(3):
            val = data[r, c]
            bg_lum = (val - vmin) / v_range if v_range > 0 else 0.5
            text_color = 'black' if 0.25 < bg_lum < 0.75 else 'white'
            ax.text(c, r, f"{val:.3f}", ha='center', va='center', 
                    fontsize=FONT_SIZES['kernel_text'], color=text_color, weight='bold')
    
    ax.set_xticks([])
    ax.set_yticks([])
    return im

# --- Main Logic ---
full_char_data = load_all_characterization_from_folder(cfg)
if not full_char_data: print("FATAL: Could not load characterization data. Exiting."); sys.exit(1)

# --- 1. SETUP ---
if KERNEL_TO_OPTIMIZE not in ALL_PRESET_KERNELS: print(f"FATAL: Kernel '{KERNEL_TO_OPTIMIZE}' not found."); sys.exit(1)
unitless_kernel = ALL_PRESET_KERNELS[KERNEL_TO_OPTIMIZE]
print(f"\nOptimizing for kernel: '{KERNEL_TO_OPTIMIZE}'\n{unitless_kernel}")
max_abs_weight = np.max(np.abs(unitless_kernel))
ideal_unitless_shape = unitless_kernel / max_abs_weight if max_abs_weight > 1e-9 else np.zeros((3,3))
min_x, min_y = min(k[0] for k in full_char_data.keys()), min(k[1] for k in full_char_data.keys())


# --- 2. OUTER LOOP: Multi-Reference V_tg Sweep ---
all_results = []
all_measured_vtgs = pd.concat([char.i_ph_vs_nd_original.index.to_series() for char in full_char_data.values()]).unique()
sorted_unique_vtgs = np.sort(all_measured_vtgs)

for ref_power_for_search in REFERENCE_ND_LEVELS_TO_TEST:
    print(f"\n===== RUNNING OPTIMIZATION WITH ND={ref_power_for_search} AS REFERENCE ======")
    
    for i, vtg_candidate in enumerate(sorted_unique_vtgs):
        if (i+1) % 20 == 0: print(f"  - Testing V_tg candidate {i+1}/{len(sorted_unique_vtgs)}")

        s_factors_for_this_vtg = []
        for r_k, c_k in np.ndindex(3, 3):
            pixel_coord = (min_x + c_k, min_y + r_k)
            char = full_char_data.get(pixel_coord)
            weight = unitless_kernel[r_k, c_k]
            if not char or abs(weight) < 1e-9: continue
            iph_at_vtg = get_iph_for_vtg_interpolated(vtg_candidate, ref_power_for_search, char)
            if (weight > 0 and iph_at_vtg > 1e-12): s_factors_for_this_vtg.append(iph_at_vtg / weight)
            elif (weight < 0 and iph_at_vtg < -1e-12): s_factors_for_this_vtg.append(abs(iph_at_vtg) / abs(weight))

        if not s_factors_for_this_vtg: continue
        s_achievable = min(s_factors_for_this_vtg)
        
        target_currents = unitless_kernel * s_achievable
        vtg_candidate_set = np.zeros((3, 3))
        for r_k, c_k in np.ndindex(3, 3):
            pixel_coord = (min_x + c_k, min_y + r_k)
            char = full_char_data.get(pixel_coord)
            if char: vtg_candidate_set[r_k, c_k] = get_vtg_for_target_iph(target_currents[r_k, c_k], char, ref_power_for_search)

        total_shape_error, achieved_currents_by_power = 0, {}
        for power in powers_to_test:
            v_tg_to_use = vtg_candidate_set
            
            achieved_currents = np.zeros((3, 3))
            for r_k, c_k in np.ndindex(3, 3):
                pixel_coord_inner = (min_x + c_k, min_y + r_k)
                char = full_char_data.get(pixel_coord_inner)
                if char: achieved_currents[r_k, c_k] = get_iph_for_vtg_interpolated(v_tg_to_use[r_k, c_k], power, char)
            
            achieved_currents_by_power[power] = achieved_currents
            max_abs_achieved = np.max(np.abs(achieved_currents))
            if max_abs_achieved < 1e-12: continue
            achieved_unitless_shape = achieved_currents / max_abs_achieved
            shape_error_for_power = np.sum((ideal_unitless_shape - achieved_unitless_shape)**2)
            total_shape_error += shape_error_for_power
            
        all_results.append({'total_loss': total_shape_error, 'candidate_vtg': vtg_candidate, 'voltages': vtg_candidate_set, 'achieved_currents': achieved_currents_by_power, 'ref_nd': ref_power_for_search})

# --- 3. SELECT WINNER AND VISUALIZE ---
if not all_results: print("FATAL: Optimization failed completely."); sys.exit(1)

sorted_results = sorted(all_results, key=lambda x: x['total_loss'])
best_result = sorted_results[0]
best_ref_nd = best_result['ref_nd']

print("\n--- FINAL OPTIMIZATION COMPLETE ---")
print(f"The absolute best voltage matrix was found when using ND={best_ref_nd} as the reference.")
print(f"Lowest Total Shape Error: {best_result['total_loss']:.4f}")
print(f"Achieved with candidate V_tg = {best_result['candidate_vtg']:.2f} V")
print("Final Best Compromise Voltage Matrix (V_tg):")
print(np.round(best_result['voltages'], 2))

# --- PLOT 1: Deep Dive of the BEST Solution (Bar Charts) ---
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
axes1_flat = axes1.flatten()
fig1.suptitle(f"Performance of Best Voltage Solution for '{KERNEL_TO_OPTIMIZE}' Kernel", fontsize=FONT_SIZES['title'])
fig1.supylabel("Normalized Weight", fontsize=FONT_SIZES['label'])
ideal_shape_flat = ideal_unitless_shape.flatten()
for i, power in enumerate(powers_to_test):
    ax = axes1_flat[i]
    achieved_currents = best_result['achieved_currents'][power]
    max_abs_achieved = np.max(np.abs(achieved_currents))
    achieved_shape_flat = (achieved_currents / max_abs_achieved if max_abs_achieved > 1e-12 else np.zeros(9)).flatten()
    rmse = np.sqrt(np.mean((ideal_shape_flat - achieved_shape_flat)**2))
    title = f"ND Filter = {power} (Shape RMSE = {rmse:.4f})"
    ax.set_title(title, fontsize=FONT_SIZES['label'])
    df = pd.DataFrame({'Ideal Shape': ideal_shape_flat, 'Achieved Shape': achieved_shape_flat})
    df.plot(kind='bar', ax=ax, color=['gray', 'royalblue'], alpha=0.8, edgecolor='black', zorder=10)
    ax.grid(True, axis='y', linestyle=':')
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['ticks'])
    ax.legend(fontsize=FONT_SIZES['legend'])
    ax.set_xticklabels([f'P({(c//3)+min_y},{(c%3)+min_x})' for c in range(9)], rotation=45, ha='right')
for i in range(len(powers_to_test), 4):
    axes1_flat[i].set_visible(False)
fig1.tight_layout(rect=[0.03, 0, 1, 0.96])
output_path1 = os.path.join(output_dir, f"{KERNEL_TO_OPTIMIZE}_performance_barchart.png")
fig1.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"Saved bar chart plot to {output_path1}")
plt.close(fig1)

# --- PLOT 2: Summary of Optimization Landscape ---
fig2, ax2 = plt.subplots(figsize=(12, 7))
colors = plt.get_cmap('viridis', len(REFERENCE_ND_LEVELS_TO_TEST))
for i, ref_nd in enumerate(REFERENCE_ND_LEVELS_TO_TEST):
    run_results = [res for res in all_results if res['ref_nd'] == ref_nd]
    if not run_results: continue
    run_results.sort(key=lambda x: x['candidate_vtg'])
    vtgs = [res['candidate_vtg'] for res in run_results]
    losses = [res['total_loss'] for res in run_results]
    ax2.plot(vtgs, losses, 'o-', color=colors(i / (len(REFERENCE_ND_LEVELS_TO_TEST)-1 if len(REFERENCE_ND_LEVELS_TO_TEST)>1 else 1) ), label=f'Search anchored at ND={ref_nd}')
ax2.plot(best_result['candidate_vtg'], best_result['total_loss'], '*', color='gold', markersize=20, markeredgecolor='black', zorder=11, label=f"Absolute Best (Loss={best_result['total_loss']:.3f})")
ax2.set_title("Optimization Landscape: Shape Error vs. Candidate Top Gate Voltage", fontsize=FONT_SIZES['title'])
ax2.set_xlabel("Candidate Top-Gate Voltage Applied Across All Pixels [V]", fontsize=FONT_SIZES['label'])
ax2.set_ylabel("Total Shape Error (Sum of SSEs)", fontsize=FONT_SIZES['label'])
ax2.tick_params(axis='both', which='major', labelsize=FONT_SIZES['ticks'])
ax2.legend(title="Optimization Run", fontsize=FONT_SIZES['legend'], title_fontsize=FONT_SIZES['legend_title'])
ax2.grid(True, linestyle=':')
fig2.tight_layout()
output_path2 = os.path.join(output_dir, f"{KERNEL_TO_OPTIMIZE}_optimization_landscape.png")
fig2.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Saved landscape plot to {output_path2}")
plt.close(fig2)

# ---  PLOT 3: Visual Comparison with New Subtitles ---
print("\n--- Generating Kernel Comparison Plots with New Subtitles ---")
global_max_abs = np.max(np.abs(ideal_unitless_shape))
vmin, vmax = -global_max_abs, global_max_abs

for power in powers_to_test:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    
    achieved_currents = best_result['achieved_currents'][power]
    max_abs_achieved = np.max(np.abs(achieved_currents))
    achieved_shape = (achieved_currents / max_abs_achieved if max_abs_achieved > 1e-12 else np.zeros((3,3)))
    
    rmse = np.sqrt(np.mean((ideal_unitless_shape.flatten() - achieved_shape.flatten())**2))
    
    title_with_error = (f"Kernel Comparison for '{KERNEL_TO_OPTIMIZE}' at ND = {power}\n"
                        f"(Shape RMSE = {rmse:.4f})")
    fig.suptitle(title_with_error, fontsize=FONT_SIZES['title'])
    
    # Define the new, more descriptive subtitles
    ideal_subtitle = "Ideal Target Shape"
    achieved_subtitle = f"Shape at ND = {power}"
    
    # Use the new subtitles in the function calls
    plot_kernel_on_ax(axes[0], ideal_unitless_shape, ideal_subtitle, vmin, vmax)
    plot_kernel_on_ax(axes[1], achieved_shape, achieved_subtitle, vmin, vmax)
    
    fig.tight_layout(rect=[0, 0, 1, 0.90]) 
    
    safe_power_str = str(power).replace('.', 'p')
    output_path = os.path.join(output_dir, f"{KERNEL_TO_OPTIMIZE}_kernel_comparison_ND_{safe_power_str}.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close(fig)

plt.show()