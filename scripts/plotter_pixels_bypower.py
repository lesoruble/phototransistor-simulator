import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import sys
from scipy.optimize import curve_fit

# --- Configuration ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_directory_path = os.path.join(project_root, "250624_SlothGUI/")
FILENAME_FILTER_KEYWORD = "KernelPhotoTopGating" 

# --- Two-stage fitting configuration ---
CLEAN_ND_FOR_PRIOR = 4
NOISY_ND_TO_FIT = 6

# --- KERNEL CONFIGURATION ---
KERNEL_3x3 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
REFERENCE_PIXEL_X = 2
REFERENCE_PIXEL_Y = 2

# --- Plotting Units ---
UNIT_SCALE = 1e9  # Convert from A to nA
UNIT_LABEL = "nA"

# --- Column Names ---
gate1_col_name, gate2_col_name = "v_bg", "v_tg"
res_mag_col, phase_col = "r_lock_in", "phase_lock_in"
iph_col, nd_filter_col = "iph_in_phase", "nd_filter"
x_coord_col, y_coord_col = "x_coord", "y_coord"

# --- FITTING MODELS & HELPERS ---
def gaussian_with_baseline(x, slope, intercept, amplitude, center, sigma):
    return (slope * x + intercept) + amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))
def lorentzian_with_baseline(x, slope, intercept, amplitude, center, gamma):
    return (slope * x + intercept) + amplitude * (gamma**2 / ((x - center)**2 + gamma**2))
def double_gaussian_with_baseline(x, slope, intercept, amp1, cen1, sig1, amp2, cen2, sig2):
    return (slope*x+intercept) + amp1*np.exp(-((x-cen1)**2)/(2*sig1**2)) + amp2*np.exp(-((x-cen2)**2)/(2*sig2**2))
def double_lorentzian_with_baseline(x, slope, intercept, amp1, cen1, gam1, amp2, cen2, gam2):
    return (slope*x+intercept) + amp1*(gam1**2/((x-cen1)**2+gam1**2)) + amp2*(gam2**2/((x-cen2)**2+gam2**2))
def piecewise_gaussian_with_baseline(x, slope1, slope2, y_at_center, amplitude, center, sigma):
    baseline = np.where(x <= center, y_at_center + slope1 * (x - center), y_at_center + slope2 * (x - center))
    return baseline + amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))
def calculate_r_squared(y_data, y_fit):
    ss_res = np.sum((y_data - y_fit)**2); ss_tot = np.sum((y_data - np.mean(y_data))**2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

# --- Loader and Helper Functions (Unchanged) ---
def find_intersections(x_data, y_data, target_y):
    intersections = [];
    for i in range(len(x_data) - 1):
        x1, y1, x2, y2 = x_data[i], y_data[i], x_data[i+1], y_data[i+1];
        if (y1 - target_y) * (y2 - target_y) < 0 and y2 - y1 != 0:
            intersections.append(x1 + (x2 - x1) * (target_y - y1) / (y2 - y1))
    return intersections
def load_special_csv(filepath):
    with open(filepath, 'r') as f: lines = f.readlines();
    header_line_index = -1;
    for i, line in enumerate(lines):
        if line.strip().startswith('# column names:'): header_line_index = i; break
    if header_line_index == -1: return pd.read_csv(filepath, comment='#', skipinitialspace=True)
    header_line = lines[header_line_index + 1].strip().lstrip('#').strip();
    header = [h.strip() for h in header_line.split(',')];
    return pd.read_csv(filepath, header=None, names=header, skiprows=header_line_index + 2, skipinitialspace=True, comment='#')

# --- Data Loading and Prep ---
all_csv_files = [f for f in os.listdir(data_directory_path) if f.lower().endswith('.csv') and FILENAME_FILTER_KEYWORD in f]
all_data_list = []
for filename in all_csv_files:
    full_file_path = os.path.join(data_directory_path, filename)
    df_run = load_special_csv(full_file_path)
    column_map = {"Gate Voltage": gate1_col_name, "Gate Voltage 2": gate2_col_name, "R Lock_in": res_mag_col, "Phase Lock_in": phase_col, "ND Filter": nd_filter_col, "X Coord": x_coord_col, "Y Coord": y_coord_col}
    df_run.rename(columns=column_map, inplace=True)
    required_cols = [gate1_col_name, gate2_col_name, res_mag_col, phase_col, nd_filter_col, x_coord_col, y_coord_col]
    if all(col in df_run.columns for col in required_cols): all_data_list.append(df_run[required_cols])
master_df = pd.concat(all_data_list, ignore_index=True)
for col in master_df.columns: master_df[col] = pd.to_numeric(master_df[col], errors="coerce")
master_df.dropna(inplace=True)
master_df[iph_col] = (master_df[res_mag_col] * np.cos(np.deg2rad(master_df[phase_col]))) * UNIT_SCALE
print(f"Total data points ready for plotting: {len(master_df)}")

# --- Main Analysis ---
unique_nd_filters = sorted(master_df[nd_filter_col].unique())
clean_df_for_priors = master_df[master_df[nd_filter_col] == CLEAN_ND_FOR_PRIOR]

# --- PART 1: LOOP FOR REGULAR ND FILTERS ---
for nd_value in unique_nd_filters:
    if int(nd_value) == NOISY_ND_TO_FIT:
        continue 

    print(f"\n--- Processing ND Filter = {int(nd_value)} ---")
    fig, ax = plt.subplots(figsize=(12, 8))
    # ... (This entire block is the original, correct logic and is unchanged) ...
    nd_group_df = master_df[master_df[nd_filter_col] == nd_value]
    unique_pixels = nd_group_df[[x_coord_col, y_coord_col]].drop_duplicates().sort_values(by=[y_coord_col, x_coord_col])
    if unique_pixels.empty: plt.close(fig); continue
    targets = {}
    ref_pixel_df = nd_group_df[(nd_group_df[x_coord_col] == REFERENCE_PIXEL_X) & (nd_group_df[y_coord_col] == REFERENCE_PIXEL_Y)]
    if not ref_pixel_df.empty:
        base_response = ref_pixel_df[iph_col].max()
        ref_pixel_weight = KERNEL_3x3[REFERENCE_PIXEL_Y - 1, REFERENCE_PIXEL_X - 1]
        if ref_pixel_weight != 0:
            unit_current = base_response / ref_pixel_weight
            for weight in np.unique(KERNEL_3x3): targets[weight] = unit_current * weight
    colors = plt.get_cmap('tab10')
    
    bottom_labels_to_plot = []
    top_label_info = None

    for i, (_, pixel_info) in enumerate(unique_pixels.iterrows()):
        x, y = pixel_info[x_coord_col], pixel_info[y_coord_col]
        sweep_df = nd_group_df[(nd_group_df[x_coord_col] == x) & (nd_group_df[y_coord_col] == y)]
        plot_data = sweep_df.groupby(gate2_col_name)[iph_col].mean().sort_index()
        if not plot_data.empty:
            num_pixels = len(unique_pixels); color = colors(i / (num_pixels - 1)) if num_pixels > 1 else colors(0.5)
            label_text = f'Pixel ({int(x)}, {int(y)})'
            ax.plot(plot_data.index, plot_data.values, marker='o', linestyle='-', label=label_text, color=color, zorder=5)
            is_reference = (x == REFERENCE_PIXEL_X and y == REFERENCE_PIXEL_Y)
            pixel_weight = KERNEL_3x3[int(y) - 1, int(x) - 1]
            if is_reference:
                vtg_at_max = plot_data.idxmax(); max_iph = plot_data.max()
                ax.plot(vtg_at_max, max_iph, 'X', color='red', markersize=7, zorder=11)
                top_label_info = {
                'x': vtg_at_max, 'text': f'{vtg_at_max:.2f}V', 'color': color,
                'line_start_y': max_iph, 'linestyle': ':'
            }            
            
            elif targets and pixel_weight in targets:
                specific_target = targets[pixel_weight]
                crossings = find_intersections(plot_data.index.values, plot_data.values, specific_target)
                if crossings:
                    ax.plot(crossings, [specific_target] * len(crossings), 'X', color='red', markersize=7, zorder=11)
                    for cross_x in crossings:
                        # Add info to our list
                        bottom_labels_to_plot.append({
                        'x': cross_x, 'text': f'{cross_x:.2f}V', 'color': color,
                        'line_start_y': specific_target, 'linestyle': '--'
                    })
                        
    y_lims = ax.get_ylim()
    x_lims = ax.get_xlim()
    fixed_line_length = (y_lims[1] - y_lims[0]) * 0.175 
    if top_label_info:
        line_end_y = top_label_info['line_start_y'] + fixed_line_length
        ax.plot([top_label_info['x'], top_label_info['x']], [top_label_info['line_start_y'], line_end_y],
                color=top_label_info['color'], linestyle=top_label_info['linestyle'], lw=1.0)
        ax.text(top_label_info['x'], line_end_y, top_label_info['text'],
                color=top_label_info['color'], ha='center', va='bottom',
                backgroundcolor=(1, 1, 1, 0.7), fontsize=14, zorder=12)

    if bottom_labels_to_plot:
        bottom_labels_to_plot.sort(key=lambda item: item['x'])
        last_x = -np.inf
        stagger_level = 0
        x_threshold = (x_lims[1] - x_lims[0]) * 0.04
        stagger_offset = (y_lims[1] - y_lims[0]) * 0.04

        for item in bottom_labels_to_plot:
            if abs(item['x'] - last_x) < x_threshold:
                stagger_level += 1
            else:
                stagger_level = 0
            
            final_y_pos = (item['line_start_y'] - fixed_line_length) - (stagger_level * stagger_offset)
            text_y_pos = line_end_y - (stagger_level * stagger_offset)

            ax.plot([item['x'], item['x']], [item['line_start_y'], final_y_pos],
                    color=item['color'], linestyle=item['linestyle'], lw=1.0)
            ax.text(item['x'], final_y_pos, item['text'],
                    color=item['color'], ha='center', va='top',
                    backgroundcolor=(1, 1, 1, 0.7), fontsize=14, zorder=12)
            
            last_x = item['x']



    if not nd_group_df.empty:
        v_bg_val = nd_group_df[gate1_col_name].iloc[0]
        ax.set_title(f"Photocurrent vs. Top Gate (ND Filter: {int(nd_value)})", fontsize=20)
    ax.set_xlabel(f"Top Gate Voltage [V]", fontsize=18); ax.set_ylabel(f"In-Phase Photocurrent [{UNIT_LABEL}]", fontsize=18)
    handles, labels = ax.get_legend_handles_labels(); by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=16)
    if targets:
        plt.draw(); xmin, xmax = ax.get_xlim()
        for weight, target_value in targets.items():
            line_color = 'green' if weight > 0 else 'blue' if weight < 0 else 'gray'
            # *** CORRECTED LOGIC FOR LINE AND TEXT PLACEMENT ***
            if weight == 8:
                line_start_x = xmin + 0.2 * (xmax - xmin) # Start line after the legend
                text_x_pos = line_start_x + 0.1 * (xmax - xmin)
            else:
                line_start_x = xmin + 0.1 * (xmax - xmin) 
                text_x_pos = xmin + 0.2 * (xmax - xmin)
            ax.plot([line_start_x, xmax], [target_value, target_value], color=line_color, linestyle='--', linewidth=1.5, zorder=10)
            ax.text(text_x_pos, target_value, f'Target (W={weight})', color=line_color, va='bottom', backgroundcolor=(1,1,1,0.7), fontsize=14)
    fig.tight_layout(); ax.grid(True)
    plt.show()

# --- PART 2: SEPARATE BLOCK FOR THE NOISY ND FILTER ---
if NOISY_ND_TO_FIT in unique_nd_filters:
    print(f"\n--- Special Processing for ND Filter = {NOISY_ND_TO_FIT} ---")
    nd_group_df = master_df[master_df[nd_filter_col] == NOISY_ND_TO_FIT]
    unique_pixels = nd_group_df[[x_coord_col, y_coord_col]].drop_duplicates().sort_values(by=[y_coord_col, x_coord_col])
    colors = plt.get_cmap('tab10')

    # STAGE 1: DIAGNOSTIC PLOTS
    print("-> Running diagnostic grid search fitting workflow...")
    fig_diag, axes_diag = plt.subplots(3, 3, figsize=(20, 15), sharex=True, sharey=True)
    fig_diag.suptitle(f'Grid Search Fit Comparison for ND={NOISY_ND_TO_FIT}', fontsize=20)
    pixel_fits = {} 
    for i, (_, pixel_info) in enumerate(unique_pixels.iterrows()):
        ax_diag = axes_diag.flat[i]; x, y = pixel_info[x_coord_col], pixel_info[y_coord_col]
        clean_data = clean_df_for_priors.groupby([x_coord_col, y_coord_col]).get_group((x,y)).groupby(gate2_col_name)[iph_col].mean().sort_index() if (x,y) in clean_df_for_priors.set_index([x_coord_col, y_coord_col]).index else pd.Series()
        noisy_data = nd_group_df[(nd_group_df[x_coord_col] == x) & (nd_group_df[y_coord_col] == y)].groupby(gate2_col_name)[iph_col].mean().sort_index()
        if noisy_data.empty: continue
        ax_diag.plot(noisy_data.index, noisy_data.values, 'ko', markersize=2, alpha=0.5, label='Raw Data')
        priors = {}
        if not clean_data.empty and len(clean_data) > 8:
            models_for_priors = {"S-Gauss": gaussian_with_baseline, "S-Lorentz": lorentzian_with_baseline, "D-Gauss": double_gaussian_with_baseline, "D-Lorentz": double_lorentzian_with_baseline}
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
        if not all_fit_results:
            ax_diag.set_title(f'Pixel ({int(x)},{int(y)}) - ALL FITS FAILED'); continue
        all_fit_results.sort(key=lambda item: item['r2'], reverse=True)
        best_fit = all_fit_results[0]
        pixel_fits[(x, y)] = best_fit
        fit_styles = {"S-Gauss": "c-", "S-Lorentz": "y-", "D-Gauss": "r--", "D-Lorentz": "b--", "P-Gauss": "m-."}
        for rank, fit_result in enumerate(all_fit_results[:3]):
            x_fit = np.linspace(noisy_data.index.min(), noisy_data.index.max(), 200)
            y_fit = fit_result['func'](x_fit, *fit_result['popt'])
            label = f"{fit_result['name']} (P:{fit_result['prior_name']}, R²={fit_result['r2']:.3f})"
            ax_diag.plot(x_fit, y_fit, fit_styles.get(fit_result['name'], 'k-'), label=label, linewidth=2.5, alpha=1-rank*0.2)
        ax_diag.set_title(f'Pixel ({int(x)},{int(y)})'); ax_diag.legend(fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.show(); plt.close(fig_diag)

    # STAGE 2: FINAL ANALYSIS PLOT
    fig_final, ax_final = plt.subplots(figsize=(12, 8))
    targets = {}
    ref_fit_info = pixel_fits.get((REFERENCE_PIXEL_X, REFERENCE_PIXEL_Y))
    if ref_fit_info and ref_fit_info.get('func') is not None:
        ref_data = nd_group_df[(nd_group_df[x_coord_col] == REFERENCE_PIXEL_X) & (nd_group_df[y_coord_col] == REFERENCE_PIXEL_Y)].groupby(gate2_col_name)[iph_col].mean().sort_index()
        x_ref = np.linspace(ref_data.index.min(), ref_data.index.max(), 500)
        y_ref_fit = ref_fit_info['func'](x_ref, *ref_fit_info['popt'])
        base_response = np.max(y_ref_fit) 
        ref_pixel_weight = KERNEL_3x3[REFERENCE_PIXEL_Y - 1, REFERENCE_PIXEL_X - 1]
        if ref_pixel_weight != 0:
            unit_current = base_response / ref_pixel_weight
            for weight in np.unique(KERNEL_3x3): targets[weight] = unit_current * weight

    for i, (_, pixel_info) in enumerate(unique_pixels.iterrows()):
        x, y = pixel_info[x_coord_col], pixel_info[y_coord_col]
        fit_info = pixel_fits.get((x, y))
        plot_data = nd_group_df[(nd_group_df[x_coord_col] == x) & (nd_group_df[y_coord_col] == y)].groupby(gate2_col_name)[iph_col].mean().sort_index()
        if plot_data.empty: continue
        
        num_pixels = len(unique_pixels); color = colors(i / (num_pixels - 1)) if num_pixels > 1 else colors(0.5)
        label_text = f'Pixel ({int(x)},{int(y)})'
        if fit_info and 'r2' in fit_info: label_text += f' (Fit: {fit_info["name"]}, R²={fit_info["r2"]:.3f})'
        ax_final.plot(plot_data.index, plot_data.values, 'o', markersize=3, alpha=0.4, color=color, label=label_text)

        if fit_info and 'func' in fit_info and fit_info['func'] is not None:
            x_fit = np.linspace(plot_data.index.min(), plot_data.index.max(), 300)
            y_fit = fit_info['func'](x_fit, *fit_info['popt'])
            ax_final.plot(x_fit, y_fit, linestyle='--', color=color, linewidth=2)
            is_reference = (x == REFERENCE_PIXEL_X and y == REFERENCE_PIXEL_Y)
            pixel_weight = KERNEL_3x3[int(y) - 1, int(x) - 1]
            if is_reference:
                vtg_at_max = x_fit[np.argmax(y_fit)]; max_iph = np.max(y_fit)
                ax_final.plot(vtg_at_max, max_iph, 'X', color='red', markersize=7, zorder=11)
                ax_final.plot([vtg_at_max, vtg_at_max], [max_iph, ax_final.get_ylim()[1]], color=color, linestyle=':', lw=1.5)
                ax_final.text(vtg_at_max, ax_final.get_ylim()[1]*0.90, f'{vtg_at_max:.2f}V', color=color, ha='center', va='bottom', fontweight='bold', backgroundcolor=(1,1,1,0.6), fontsize=14)
            elif targets and pixel_weight in targets:
                specific_target = targets[pixel_weight]
                crossings = find_intersections(x_fit, y_fit, specific_target)
                if crossings:
                    ax_final.plot(crossings, [specific_target] * len(crossings), 'X', color='red', markersize=7, zorder=11)
                    for cross_x in crossings:
                        ax_final.plot([cross_x, cross_x], [ax_final.get_ylim()[0]*0.9, specific_target], color=color, linestyle='--', lw=1)
                        ax_final.text(cross_x, ax_final.get_ylim()[0]*0.88, f'{cross_x:.2f}V', color=color, ha='center', va='top', backgroundcolor=(1,1,1,0.6))
    
    v_bg_val = nd_group_df[gate1_col_name].iloc[0]
    ax_final.set_title(f"Final Analysis for ND={NOISY_ND_TO_FIT} using Best Fits", fontsize=20)
    ax_final.set_xlabel(f"Top Gate Voltage [V]", fontsize=18); ax_final.set_ylabel(f"In-Phase Photocurrent [{UNIT_LABEL}]", fontsize=18)
    
    handles, labels = ax_final.get_legend_handles_labels(); by_label = dict(zip(labels, handles))
    ax_final.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=14)

    if targets:
        plt.draw()
        xmin, xmax = ax_final.get_xlim()
        for weight, target_value in targets.items():
            line_color = 'green' if weight > 0 else 'blue' if weight < 0 else 'gray'

            if weight == 8:
                line_start_x = xmin + 0.2 * (xmax - xmin) 
                text_x_pos = line_start_x + 0.1 * (xmax - xmin)
            else:
                line_start_x = xmin + 0.1 * (xmax - xmin) 
                text_x_pos = xmin + 0.2 * (xmax - xmin)
            
            ax_final.plot([line_start_x, xmax], [target_value, target_value], color=line_color, linestyle='--', linewidth=1.5, zorder=10)
            ax_final.text(text_x_pos, target_value, f'Target (W={weight})', color=line_color, va='bottom', backgroundcolor=(1,1,1,0.7), fontsize=14)
    
    fig_final.tight_layout(); ax_final.grid(True)
    plt.show()

print("\nAnalysis complete.")