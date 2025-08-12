import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import griddata
from pathlib import Path
import sys

# --- Configuration ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_directory_path = os.path.join(project_root, "250624_SlothGUI/")
FILENAME_FILTER_KEYWORD = "KernelDoublePhotoGating"
ARRAY_ROWS = 3
ARRAY_COLS = 3

# --- Analysis Parameters ---
decimals = 2
UNIT_SCALE = 1e9  # To convert from Amps to microamps (µA)
UNIT_LABEL = "nA"

# --- NEW: Font Size Configuration for Plots ---
FONT_SIZES = {
    'title': 20,
    'label': 18,
    'legend': 15,
    'ticks': 18
}

# --- Column Names (Internal Standard) ---
gate1_col_name = "gate_voltage"
gate2_col_name = "gate_voltage_2"
res_mag_col    = "r_lock_in"
phase_col      = "phase_lock_in"
x_coord_col    = "X Coord"
y_coord_col    = "Y Coord"

# --- Smart CSV Loader to handle different formats ---
def load_special_csv(filepath):
    """
    Loads a CSV file that may contain commented header lines.
    It automatically finds the real header and the start of the data.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    header_line_index, data_start_index, header = -1, 0, None
    for i, line in enumerate(lines):
        if line.strip().startswith('# column names:'):
            header_line_index = i + 1
            break
    if header_line_index != -1:
        header_str = lines[header_line_index].strip().lstrip('#').strip()
        header = [h.strip() for h in header_str.split(',')]
        data_start_index = header_line_index + 1
    else:
        data_start_index = 1
        header = [h.strip() for h in lines[0].strip().split(',')]
    df = pd.read_csv(filepath, header=None, names=header, skiprows=data_start_index,
                     skipinitialspace=True, comment='#')
    return df

# --- Helper Function for Heatmap Data Generation ---
def generate_heatmap_data(df, x_col, y_col, z_col):
    x_data, y_data, z_data = df[x_col], df[y_col], df[z_col]
    unique_x, unique_y = np.unique(x_data), np.unique(y_data)
    if len(unique_x) < 2 or len(unique_y) < 2: return None, None, None
    x_grid, y_grid = np.meshgrid(unique_x, unique_y)
    try:
        z_grid = griddata((x_data, y_data), z_data, (x_grid, y_grid), method='linear')
        return x_grid, y_grid, z_grid
    except Exception as e:
        print(f"  Warning: Griddata interpolation failed: {e}")
        return None, None, None

# --- Discover and Filter CSV files ---
try:
    if not os.path.isdir(data_directory_path):
        raise FileNotFoundError(f"Directory not found: '{data_directory_path}'")
    all_csv_files = [f for f in os.listdir(data_directory_path) if f.lower().endswith('.csv') and FILENAME_FILTER_KEYWORD in f]
    if not all_csv_files:
        print(f"Error: No CSV files containing '{FILENAME_FILTER_KEYWORD}' found in '{data_directory_path}'")
        exit()
    print(f"Found {len(all_csv_files)} files matching '{FILENAME_FILTER_KEYWORD}'.\n" + "-"*20)
except Exception as e:
    print(f"Error accessing directory: {e}"); exit()

# --- Storage for results ---
results_by_filename, all_swing_series, all_dataframes = {}, [], {}

# =============================================================================
# --- LOOP 1: PROCESS ALL FILES AND GATHER DATA ---
# =============================================================================
print("--- Processing all files to gather data and find best overall bottom-gate voltage ---\n")
for filename in all_csv_files:
    print(f"Processing: {filename}...")
    full_file_path = os.path.join(data_directory_path, filename)
    try:
        df_run = load_special_csv(full_file_path)
        column_map = {"Gate Voltage": gate1_col_name, "Gate Voltage 2": gate2_col_name, "R Lock_in": res_mag_col, "Phase Lock_in": phase_col}
        df_run.rename(columns=column_map, inplace=True)
    except Exception as e:
        print(f"  ! Warning: Could not read or process file. Error: {e}. Skipping."); continue

    cols_to_check = [gate1_col_name, gate2_col_name, res_mag_col, phase_col, x_coord_col, y_coord_col]
    if not all(col in df_run.columns for col in cols_to_check):
        print(f"  ! Error: One or more required columns not found. Skipping."); continue

    for col in cols_to_check: df_run[col] = pd.to_numeric(df_run[col], errors="coerce")
    df_run.dropna(subset=cols_to_check, inplace=True)
    if df_run.empty or df_run[gate1_col_name].nunique() < 2 or df_run[gate2_col_name].nunique() < 2:
        print(f"  ! Insufficient data for analysis. Skipping."); continue

    x_coord, y_coord = int(df_run[x_coord_col].iloc[0]), int(df_run[y_coord_col].iloc[0])
    df_run[gate1_col_name], df_run[gate2_col_name] = df_run[gate1_col_name].round(decimals), df_run[gate2_col_name].round(decimals)
    df_run["iph_in_phase"] = df_run[res_mag_col] * np.cos(np.deg2rad(df_run[phase_col])) * UNIT_SCALE

    initial_rows = len(df_run)
    df_run = df_run.groupby([gate1_col_name, gate2_col_name]).mean().reset_index()
    averaged_rows = len(df_run)
    if initial_rows > averaged_rows:
        print(f"  - Averaged {initial_rows - averaged_rows} duplicate V_bg/V_tg points down to {averaged_rows} unique points.")
    
    all_dataframes[filename] = df_run.copy()

    vbg_groups = df_run.groupby(gate1_col_name)
    iph_x_range_per_vbg = vbg_groups["iph_in_phase"].apply(lambda x: x.max() - x.min() if len(x.unique()) > 1 else 0)

    if iph_x_range_per_vbg.empty or iph_x_range_per_vbg.max() == 0:
        max_swing_vbg, max_swing_value = np.nan, 0
        print(f"  - No meaningful photocurrent swing found.")
    else:
        max_swing_vbg, max_swing_value = iph_x_range_per_vbg.idxmax(), iph_x_range_per_vbg.max()
        print(f"  - Pixel ({x_coord}, {y_coord}): Maximum swing of {max_swing_value:.2e} A at Bottom-Gate Voltage = {max_swing_vbg:.{decimals}f} V")

    results_by_filename[filename] = {"x_coord": x_coord, "y_coord": y_coord, "max_range_value": max_swing_value, "max_range_vbg": max_swing_vbg, "swing_series": iph_x_range_per_vbg}
    all_swing_series.append(iph_x_range_per_vbg.rename(f"Pixel ({x_coord}, {y_coord})"))

if not all_dataframes: print("\nNo data was successfully processed. Exiting."); exit()

print("\n--- Combined Analysis: Finding Best Bottom-Gate Voltage Across All Pixels ---")
combined_swings_df = pd.concat(all_swing_series, axis=1)
mean_swing_per_vbg = combined_swings_df.mean(axis=1)
best_overall_vbg = mean_swing_per_vbg.idxmax()
best_mean_swing_value = mean_swing_per_vbg.max()
print(f"Best Overall Bottom-Gate Voltage (highest average swing): {best_overall_vbg:.{decimals}f} V")

# =============================================================================
# --- LOOP 2: GENERATE INDIVIDUAL PLOTS ---
# =============================================================================
print("\n--- Generating Individual Plots for Each Pixel ---")
for filename, result in results_by_filename.items():
    px, py = result['x_coord'], result['y_coord']
    print(f"-> Plotting for Pixel ({px}, {py})...")
    plt.figure(figsize=(10, 6))
    plt.plot(result['swing_series'].index, result['swing_series'].values, marker='o')
    plt.axvline(result['max_range_vbg'], color='red', linestyle=':', lw=2, label=f'Pixel Best Bottom-Gate Voltage = {result["max_range_vbg"]:.{decimals}f} V')
    plt.axvline(best_overall_vbg, color='green', linestyle='--', lw=2, label=f'Overall Best Bottom-Gate Voltage = {best_overall_vbg:.{decimals}f} V')
    plt.title(f"Photocurrent Swing vs. Bottom-Gate Voltage for Pixel ({px}, {py})", fontsize=FONT_SIZES['title'])
    plt.xlabel("Bottom-Gate Voltage [V]", fontsize=FONT_SIZES['label'])
    plt.ylabel(f"Photocurrent Swing [{UNIT_LABEL}]", fontsize=FONT_SIZES['label'])
    plt.xticks(fontsize=FONT_SIZES['ticks'])
    plt.yticks(fontsize=FONT_SIZES['ticks'])
    plt.legend(fontsize=FONT_SIZES['legend'])
    plt.grid(True, which='both', linestyle='--'); plt.tight_layout()

    x_g, y_g, z_g = generate_heatmap_data(all_dataframes[filename], gate1_col_name, gate2_col_name, "iph_in_phase")
    if x_g is not None:
        plt.figure(figsize=(9, 7))
        abs_max = np.nanmax(np.abs(z_g))
        pcm = plt.pcolormesh(x_g, y_g, z_g, shading='auto', cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
        cbar = plt.colorbar(pcm, label="In-Phase Photocurrent [mA]")
        cbar.ax.tick_params(labelsize=FONT_SIZES['ticks'])
        cbar.set_label(f"In-Phase Photocurrent [{UNIT_LABEL}]", size=FONT_SIZES['label'])
        plt.axvline(result['max_range_vbg'], color='red', linestyle=':', lw=2.5, label=f'Pixel Best Bottom-Gate Voltage = {result["max_range_vbg"]:.{decimals}f} V')
        plt.axvline(best_overall_vbg, color='green', linestyle='--', lw=2.5, label=f'Overall Best Bottom-Gate Voltage = {best_overall_vbg:.{decimals}f} V')
        plt.title(f"Photocurrent Map for Pixel ({px}, {py})", fontsize=FONT_SIZES['title'])
        plt.xlabel("Bottom-Gate Voltage [V]", fontsize=FONT_SIZES['label'])
        plt.ylabel("Top-Gate Voltage [V]", fontsize=FONT_SIZES['label'])
        plt.xticks(fontsize=FONT_SIZES['ticks'])
        plt.yticks(fontsize=FONT_SIZES['ticks'])
        plt.legend(fontsize=FONT_SIZES['legend'])
        plt.tight_layout()

# =============================================================================
# --- SWING CALCULATION DEMONSTRATION PLOT ---
# =============================================================================
print("\n--- Generating Swing Calculation Demonstration for Pixel (2, 2) ---")
target_pixel_filename = None
for filename, result in results_by_filename.items():
    if result['x_coord'] == 2 and result['y_coord'] == 2:
        target_pixel_filename = filename; target_pixel_results = result; break
if target_pixel_filename:
    px, py = target_pixel_results['x_coord'], target_pixel_results['y_coord']
    vbg_for_max_swing = target_pixel_results['max_range_vbg']
    df_target_pixel = all_dataframes[target_pixel_filename]
    slice_data = df_target_pixel[np.isclose(df_target_pixel[gate1_col_name], vbg_for_max_swing)].copy()
    slice_data.sort_values(by=gate2_col_name, inplace=True)
    if not slice_data.empty:
        iph_max, iph_min = slice_data['iph_in_phase'].max(), slice_data['iph_in_phase'].min()
        swing_value = iph_max - iph_min
        
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.plot(slice_data[gate2_col_name], slice_data['iph_in_phase'], marker='o', label='Photocurrent Data')
        ax.axhline(iph_max, color='green', linestyle='--', label=f'Maximum Photocurrent = {iph_max:.2f}{UNIT_LABEL}')
        ax.axhline(iph_min, color='red', linestyle='--', label=f'Minimum Photocurrent = {iph_min:.2f}{UNIT_LABEL}')
        ax.text(0.95, 0.5, f'Swing = {swing_value:.2f}{UNIT_LABEL}', transform=ax.transAxes, va='center', ha='right', fontsize=14, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        ax.set_title(f"Pixel ({px}, {py}) at fixed bottom-gate voltage ({vbg_for_max_swing:.2f}V)", fontsize=FONT_SIZES['title'])
        ax.set_xlabel("Top Gate Voltage [V]", fontsize=FONT_SIZES['label'])
        ax.set_ylabel(f"In-Phase Photocurrent [{UNIT_LABEL}]", fontsize=FONT_SIZES['label'])
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['ticks'])
        ax.legend(fontsize=FONT_SIZES['legend']); ax.grid(True, linestyle=':')
        plt.tight_layout()

# =============================================================================
# --- FINAL SUMMARY AND COMPARISON PLOTS ---
# =============================================================================
print("\n--- Generating Summary & Comparison Plots ---")
# --- Summary Table ---
sorted_results_list = sorted(results_by_filename.values(), key=lambda item: item["max_range_value"], reverse=True)
print("\n--- Summary of Maximum Swing per Pixel (Sorted by Performance) ---")
print(f"{'Pixel':<10} | {'Max Swing ('+UNIT_LABEL+')':<15} | {'Bottom-Gate V for Max Swing (V)'}"); print("-" * 60)
for res in sorted_results_list:
    if res['max_range_value'] > 0: print(f"({res['x_coord']}, {res['y_coord']}){'':<4} | {res['max_range_value']:.3f} {'':<11}| {res['max_range_vbg']:.{decimals}f}")

# --- Plot: Spatial Performance Map ---
performance_grid = np.full((ARRAY_ROWS, ARRAY_COLS), np.nan)
for res in results_by_filename.values():
    row_idx, col_idx = res['y_coord'] - 1, res['x_coord'] - 1
    if 0 <= row_idx < ARRAY_ROWS and 0 <= col_idx < ARRAY_COLS: performance_grid[row_idx, col_idx] = res['max_range_value']
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(performance_grid, cmap='viridis', interpolation='nearest')
ax.set_xticks(np.arange(ARRAY_COLS)); ax.set_yticks(np.arange(ARRAY_ROWS))
ax.set_xticklabels(np.arange(1, ARRAY_COLS + 1), fontsize=FONT_SIZES['ticks'])
ax.set_yticklabels(np.arange(1, ARRAY_ROWS + 1), fontsize=FONT_SIZES['ticks'])
ax.set_xlabel("X Coordinate", fontsize=FONT_SIZES['label']); ax.set_ylabel("Y Coordinate", fontsize=FONT_SIZES['label'])
ax.set_title(f"Maximum Photocurrent Swing Across {ARRAY_ROWS}x{ARRAY_COLS} Array", pad=20, fontsize=FONT_SIZES['title'])
for i in range(ARRAY_ROWS):
    for j in range(ARRAY_COLS):
        if not np.isnan(performance_grid[i, j]):
            ax.text(j, i, f"{performance_grid[i, j]:.2f}", ha="center", va="center", color="w", weight="bold")
cbar = fig.colorbar(im, label="Maximum Photocurrent Swing [A]")
cbar.ax.tick_params(labelsize=FONT_SIZES['ticks'])
cbar.set_label(f"Maximum Photocurrent Swing [{UNIT_LABEL}]", size=FONT_SIZES['label'])

# --- Plot: Combined Individual and Mean Swing Comparison ---
plt.figure(figsize=(12, 8))
for series in all_swing_series:
    plt.plot(series.index, series.values, linestyle='-', marker=None, lw=2.0, alpha=0.7, label=series.name)
plt.plot(mean_swing_per_vbg.index, mean_swing_per_vbg.values, linestyle='--', color='k', marker='.', markersize=4, lw=1.5, label='Mean Swing', zorder=10)
plt.plot(best_overall_vbg, best_mean_swing_value, marker='*', color='gold', markersize=18, markeredgecolor='black', label=f'Max of Mean ({best_mean_swing_value:.2f} {UNIT_LABEL})', zorder=11, linestyle='none')
plt.axvline(best_overall_vbg, color='r', linestyle='--', lw=2, label=f'Overall Best Bottom-Gate Voltage = {best_overall_vbg:.{decimals}f} V')
plt.title("Individual and Mean Photocurrent Swings vs. Bottom-Gate Voltage", fontsize=FONT_SIZES['title'])
plt.xlabel("Bottom-Gate Voltage [V]", fontsize=FONT_SIZES['label'])
plt.ylabel(f"Photocurrent Swing [{UNIT_LABEL}]", fontsize=FONT_SIZES['label'])
plt.xticks(fontsize=FONT_SIZES['ticks'])
plt.yticks(fontsize=FONT_SIZES['ticks'])
plt.legend(loc='best', title="Pixels & Mean", fontsize=FONT_SIZES['legend']); plt.grid(True, which='both', linestyle='--'); plt.tight_layout()

# --- Plot: Sequential Heatmap Subplots ---
num_files = len(all_dataframes)
if num_files > 0:
    ncols = 3 if num_files > 2 else num_files; nrows = int(np.ceil(num_files / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False, constrained_layout=True)
    axes = axes.flatten()
    global_max = max(df["iph_in_phase"].abs().max() for df in all_dataframes.values())
    pcm = None
    for i, (filename, df) in enumerate(all_dataframes.items()):
        ax = axes[i]; result = results_by_filename[filename]; px, py = result['x_coord'], result['y_coord']
        x_g, y_g, z_g = generate_heatmap_data(df, gate1_col_name, gate2_col_name, "iph_in_phase")
        if x_g is not None:
            pcm = ax.pcolormesh(x_g, y_g, z_g, shading='auto', cmap='RdBu_r', vmin=-global_max, vmax=global_max)
            ax.axvline(best_overall_vbg, color='green', linestyle='--', lw=2.5)
            ax.set_title(f"Pixel ({px}, {py})", fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=FONT_SIZES['ticks'] - 2)
        else:
            ax.text(0.5, 0.5, 'Grid Error', ha='center', va='center'); ax.set_title(f"Pixel ({px}, {py})", fontsize=10)
    for i in range(num_files, len(axes)): axes[i].set_visible(False)
    if pcm:
        cbar = fig.colorbar(pcm, ax=axes.tolist(), label="In-Phase Photocurrent [A]", aspect=40, pad=0.02)
        cbar.ax.tick_params(labelsize=FONT_SIZES['ticks'])
        cbar.set_label(f"In-Phase Photocurrent [{UNIT_LABEL}]", size=FONT_SIZES['label'])

print("\nOverall analysis script finished.")
plt.show() # Show all figures at the end