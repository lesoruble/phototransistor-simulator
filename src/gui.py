# src/gui.py

from __future__ import annotations
__all__ = ["launch_gui", "ConvolutionApp"]

import tkinter as tk
from tkinter import ttk, filedialog, font as tkFont
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import logging
import json
from scipy.optimize import curve_fit
import os

from . import image_comparator
from .config import KernelConfig
from .data_io import load_all_characterization_from_folder, normalize_input_image, SimpleChar
from .kernel import get_iph_for_vtg_interpolated
from .presets import ALL_PRESET_KERNELS, ZX_SPECTRUM_FONT
from .simulator import Simulator, FixedPowerResult, DynamicModeResult


log = logging.getLogger(__name__)
cfg = KernelConfig()

# --- Module-level Globals ---
# These variables store the state of the kernel configuration and convolution process.
kernel_cells = np.empty((3, 3), dtype=object) # Stores GUI widgets for each kernel cell
current_kernel_weights, current_kernel_voltages = np.zeros((3, 3)), np.zeros((3, 3))
fixed_optimal_voltages = np.zeros((3, 3)) # Voltages for Dynamic Mode
input_image_np = None # The raw input image as a NumPy array

# State for step-by-step convolution
convolution_active, current_conv_r, current_conv_c = False, 0, 0
output_image_step_by_step_amps, padded_input_image_for_step_normalized = None, None

# Stored results from a full convolution
raw_convolved_output_amps = None
normalized_grayscale_output = None

# Debugging flag
enable_debug_write_for_first_patch = False


class PlotWindow(tk.Toplevel):
    """A simple Toplevel window for displaying a Matplotlib figure."""
    def __init__(self, parent, fig, title="Plot"):
        super().__init__(parent)
        self.title(title)
        self.geometry("900x650")
        
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add a close button
        close_button = ttk.Button(self, text="Close", command=self.destroy)
        close_button.pack(side=tk.BOTTOM, pady=5)

class SaveOptionsDialog(tk.Toplevel):
    """A custom dialog to ask the user how they want to save the CSV data."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Select Save Format")
        self.transient(parent) # Keep dialog on top of the main window
        self.result = None # This will store the user's choice

        # --- Setup Widgets ---
        self.choice_var = tk.StringVar(value="raw") # Default to 'raw'

        ttk.Label(self, text="Please choose the format for the CSV file:").pack(padx=20, pady=10)

        main_frame = ttk.Frame(self)
        main_frame.pack(padx=20, pady=10, fill="x")

        ttk.Radiobutton(
            main_frame,
            text="Raw Physical Currents (Amps)",
            variable=self.choice_var,
            value="raw"
        ).pack(anchor="w", pady=5)

        ttk.Radiobutton(
            main_frame,
            text="Normalized Grayscale Values (0-255)",
            variable=self.choice_var,
            value="grayscale"
        ).pack(anchor="w", pady=5)
        
        # --- OK and Cancel Buttons ---
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="OK", command=self.on_ok).pack(side="left", padx=10)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side="left", padx=10)

        # Make the dialog modal
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.wait_window(self)

    def on_ok(self):
        self.result = self.choice_var.get()
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()

# --- End of New Dialog Class ---


class ConvolutionApp:
    def __init__(self, master):
        self.master = master; master.title("Adaptive Phototransistor Kernel Simulator"); self.detail_font = tkFont.Font(family="Courier New", size=14); self.output_image_tk_ref, self.input_image_tk_ref = None, None
        self.current_preset_name = "CUSTOM"
        self.show_plots_var = tk.BooleanVar(value=True) # On by default 
        self.full_char_data = load_all_characterization_from_folder(cfg)
        self.simulator = Simulator(self.full_char_data, cfg) if self.full_char_data else None

        # --- Load pre-computed voltage cache for performance ---
        self.nd5_voltage_cache = None
        try:
            with open("nd5_voltage_cache.json", 'r') as f:
                self.nd5_voltage_cache = json.load(f)
            log.info("Successfully loaded 'nd5_voltage_cache.json'. ND=5 will be fast.")
        except FileNotFoundError:
            log.warning("'nd5_voltage_cache.json' not found. ND=5 calculations will be disabled.")
        except Exception as e:
            log.error(f"Error loading cache file: {e}")

        # --- Initialize application state based on data availability ---
        if self.full_char_data:
            first_pixel_char = next(iter(self.full_char_data.values())); self.power_levels_available = first_pixel_char.nd_axis
            
            # Exclude certain power levels if they are problematic or not needed
            if 6.0 in self.power_levels_available:
                self.power_levels_available = np.delete(self.power_levels_available, np.where(self.power_levels_available == 6.0))
                log.info("ND=6 has been excluded from the analysis.")
            
            if self.nd5_voltage_cache is None and 5.0 in self.power_levels_available:
                self.power_levels_available = np.delete(self.power_levels_available, np.where(self.power_levels_available == 5.0))
        
        # --- Build the GUI ---
        self.setup_main_layout()
        
        # --- Set initial GUI state ---
        if self.simulator:
            self.power_combobox['values'] = ["Dynamic Mode"] + self.power_levels_available.tolist()
            self.power_combobox.set("Dynamic Mode")
            self.vis_mode_combo.set("Grayscale")
            self.apply_preset_kernel("IDENTITY")
            self.load_default_vertical_line()
        else:
            self.update_step_text("FATAL: Could not load data or initialize simulator.")
            for child in self.content_main_frame.winfo_children():
                if isinstance(child, ttk.Frame):
                    for widget in child.winfo_children():
                        if isinstance(widget, (ttk.Button, ttk.Entry, ttk.Combobox)):
                            widget.config(state=tk.DISABLED)
            self.load_image_button.config(state=tk.NORMAL)

            
    def is_in_dynamic_mode(self): return self.power_combobox.get() == "Dynamic Mode"

    def setup_main_layout(self):
        # Main scrollable canvas setup
        self.canvas = tk.Canvas(self.master); self.scrollbar_y = ttk.Scrollbar(self.master, orient="vertical", command=self.canvas.yview); self.scrollbar_x = ttk.Scrollbar(self.master, orient="horizontal", command=self.canvas.xview); self.scrollable_frame = ttk.Frame(self.canvas); self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))); self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw"); self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set); self.canvas.grid(row=0, column=0, sticky="nsew"); self.scrollbar_y.grid(row=0, column=1, sticky="ns"); self.scrollbar_x.grid(row=1, column=0, sticky="ew"); self.master.grid_rowconfigure(0, weight=1); self.master.grid_columnconfigure(0, weight=1)
        self.content_main_frame = ttk.Frame(self.scrollable_frame, padding="10"); self.content_main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Define all major frames first ---
        self.row0_frame = ttk.Frame(self.content_main_frame); self.row0_frame.pack(fill=tk.X, pady=5) 
        self.row1_frame = ttk.Frame(self.content_main_frame); self.row1_frame.pack(fill=tk.X, pady=10)
        self.conv_control_outer_frame = ttk.LabelFrame(self.content_main_frame, text="Convolution Control"); self.conv_control_outer_frame.pack(fill=tk.X, pady=5, padx=5) 
        self.bottom_row_frame = ttk.Frame(self.content_main_frame); self.bottom_row_frame.pack(fill=tk.BOTH, expand=True, pady=10)

         # --- Populate Bottom Row (Display Panels) ---
        self.input_display_frame = ttk.Frame(self.bottom_row_frame)
        ttk.Label(self.input_display_frame, text="Input Image").pack()
        self.input_fig = Figure(figsize=(4, 4), dpi=100)
        self.input_ax = self.input_fig.add_subplot(111)
        self.input_canvas = FigureCanvasTkAgg(self.input_fig, master=self.input_display_frame)
        
        self.output_display_frame = ttk.Frame(self.bottom_row_frame)
        ttk.Label(self.output_display_frame, text="Convolved Output").pack()
        self.output_fig = Figure(figsize=(4, 4), dpi=100)
        self.output_ax = self.output_fig.add_subplot(111)
        self.output_canvas = FigureCanvasTkAgg(self.output_fig, master=self.output_display_frame)

        self.animation_frame = ttk.LabelFrame(self.bottom_row_frame, text="Convolution Step Details")
        self.convolution_step_text = tk.Text(self.animation_frame, height=20, width=60, wrap=tk.WORD, font=self.detail_font, relief="sunken", borderwidth=1)
        self.convolution_step_text.insert(tk.END, "Initializing..."); self.convolution_step_text.config(state=tk.DISABLED)

        # Pack them after creation
        self.input_display_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.input_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.output_display_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.output_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.animation_frame.pack(side=tk.RIGHT, padx=10, pady=5, fill=tk.Y, expand=False)
        self.convolution_step_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        # --- End Display Panel Creation ---

        # --- Row 0: Image Loading, Device Control, Presets ---
        self.image_controls_frame = ttk.LabelFrame(self.row0_frame, text="Default Images"); self.image_controls_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.NONE)
        top_row_controls = ttk.Frame(self.image_controls_frame); top_row_controls.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(2, 4))
        self.load_image_button = ttk.Button(top_row_controls, text="Load File...", command=self.load_image); self.load_image_button.pack(side=tk.LEFT, padx=2)
        self.default_vert_line_button = ttk.Button(top_row_controls, text="Vert L", command=self.load_default_vertical_line); self.default_vert_line_button.pack(side=tk.LEFT, padx=2)
        self.default_horiz_line_button = ttk.Button(top_row_controls, text="Horiz L", command=self.load_default_horizontal_line); self.default_horiz_line_button.pack(side=tk.LEFT, padx=2)
        self.default_dot_button = ttk.Button(top_row_controls, text="Dot", command=self.load_default_center_dot); self.default_dot_button.pack(side=tk.LEFT, padx=2)
        bottom_row_controls = ttk.Frame(self.image_controls_frame); bottom_row_controls.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)
        letter_frame = ttk.Frame(bottom_row_controls); letter_frame.pack(side=tk.LEFT, padx=0)
        ttk.Label(letter_frame, text="ZX Font:").pack(side=tk.LEFT, padx=(0,2))
        self.letter_combobox = ttk.Combobox(letter_frame, width=4, state="readonly", values=sorted(list(ZX_SPECTRUM_FONT.keys()))); self.letter_combobox.pack(side=tk.LEFT, padx=2); self.letter_combobox.set("A")
        self.load_letter_button = ttk.Button(letter_frame, text="Load", width=5, command=lambda: self._load_zx_spectrum_char(self.letter_combobox.get())); self.load_letter_button.pack(side=tk.LEFT, padx=2)
        icon_frame = ttk.Frame(bottom_row_controls); icon_frame.pack(side=tk.LEFT, padx=5)
        ttk.Label(icon_frame, text="Icons:").pack(side=tk.LEFT, padx=(5,2))
        self.eye_button = ttk.Button(icon_frame, text="Eye", width=5, command=self.load_default_eye); self.eye_button.pack(side=tk.LEFT, padx=2)
        self.sword_button = ttk.Button(icon_frame, text="Sword", width=5, command=self.load_default_sword); self.sword_button.pack(side=tk.LEFT, padx=2)
        self.blitz_button = ttk.Button(icon_frame, text="Blitz", width=5, command=self.load_default_blitz); self.blitz_button.pack(side=tk.LEFT, padx=2)
        self.power_frame = ttk.LabelFrame(self.row0_frame, text="Device Control"); self.power_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.Y)
        ttk.Label(self.power_frame, text="Optical Power (ND):").pack(pady=(5,2))
        self.power_combobox = ttk.Combobox(self.power_frame, state="readonly", width=12); self.power_combobox.pack(pady=(0,5), padx=5); self.power_combobox.bind("<<ComboboxSelected>>", self.on_power_selected)
        self.presets_frame = ttk.LabelFrame(self.row0_frame, text="Preset Kernels (Unitless)"); self.presets_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.X, expand=True)
        for idx, kernel_name in enumerate(ALL_PRESET_KERNELS.keys()): btn = ttk.Button(self.presets_frame, text=kernel_name, width=12, command=lambda kn=kernel_name: self.apply_preset_kernel(kn)); btn.grid(row=idx // 3, column=idx % 3, padx=1, pady=1, sticky="ew")
        
        # --- Row 1: Kernel Configuration and Visualization ---
        self.kernel_config_frame = ttk.LabelFrame(self.row1_frame, text="Kernel Cell Config"); self.kernel_config_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y, anchor='n')
        for r_idx in range(3):
            for c_idx in range(3):
                cell_frame = ttk.Frame(self.kernel_config_frame, borderwidth=1, relief="sunken", padding=3); cell_frame.grid(row=r_idx, column=c_idx, padx=2, pady=2, sticky="nsew")
                ttk.Label(cell_frame, text=f"W (P{r_idx}{c_idx}):").grid(row=0, column=0, sticky=tk.W)
                desired_w_entry = ttk.Entry(cell_frame, width=8); desired_w_entry.insert(0, "0.0"); desired_w_entry.grid(row=0, column=1, sticky=tk.EW, padx=2)
                desired_w_entry.bind("<KeyRelease>", self.handle_manual_weight_entry)
                found_vtg_label = ttk.Label(cell_frame, text="V_TG: N/A", width=20); found_vtg_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(1,0))
                effective_nd_label = ttk.Label(cell_frame, text="Eff. ND: N/A", width=20); effective_nd_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(1,0))
                actual_iph_label = ttk.Label(cell_frame, text="I_ph: N/A", width=20); actual_iph_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(1,0))
                kernel_cells[r_idx,c_idx] = {'desired_w_entry': desired_w_entry, 'found_vtg_label': found_vtg_label, 'effective_nd_label': effective_nd_label, 'actual_iph_label': actual_iph_label}
        self.update_kernel_button = ttk.Button(self.kernel_config_frame, text="Update/Optimize", command=self.update_all_kernel_cells_from_desired_weights); self.update_kernel_button.grid(row=4, column=0, columnspan=3, pady=5)
        self.kernel_display_frame = ttk.Frame(self.row1_frame); self.kernel_display_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True, anchor='n')
        ttk.Label(self.kernel_display_frame, text="Kernel Gate Voltages (V_TG) [V]").pack()
        self.kernel_fig, self.kernel_ax = plt.subplots(figsize=(2.2, 2.2)); self.kernel_canvas_widget = FigureCanvasTkAgg(self.kernel_fig, master=self.kernel_display_frame); self.kernel_canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # --- Convolution Control Frame ---
        conv_control_frame_top = ttk.Frame(self.conv_control_outer_frame); conv_control_frame_top.pack(fill=tk.X, expand=True, pady=(5, 2), padx=5)
        conv_control_frame_bottom = ttk.Frame(self.conv_control_outer_frame); conv_control_frame_bottom.pack(fill=tk.X, expand=True, pady=(2, 5), padx=5)
        self.convolve_full_button = ttk.Button(conv_control_frame_top, text="Convolve Full Image", command=self.convolve_full_image_and_display); self.convolve_full_button.pack(side=tk.LEFT, padx=5)
        self.reset_step_button = ttk.Button(conv_control_frame_top, text="Start/Reset Step-by-Step", command=self.start_reset_step_by_step); self.reset_step_button.pack(side=tk.LEFT, padx=5)
        self.next_step_button = ttk.Button(conv_control_frame_top, text="Next Step", command=self.perform_one_convolution_step, state=tk.DISABLED); self.next_step_button.pack(side=tk.LEFT, padx=5)
        self.five_steps_button = ttk.Button(conv_control_frame_top, text="5 Steps", command=lambda: self.perform_n_convolution_steps(5), state=tk.DISABLED); self.five_steps_button.pack(side=tk.LEFT, padx=5)
        self.finish_steps_button = ttk.Button(conv_control_frame_top, text="Finish Steps", command=self.finish_all_convolution_steps, state=tk.DISABLED); self.finish_steps_button.pack(side=tk.LEFT, padx=5)
        self.plot_response_button = ttk.Button(conv_control_frame_bottom, text="Plot Dynamic Response", command=self.plot_dynamic_response, state=tk.DISABLED); self.plot_response_button.pack(side=tk.LEFT, padx=5)
        self.save_csv_button = ttk.Button(conv_control_frame_bottom, text="Save CSV...", command=self.save_output_to_csv, state=tk.DISABLED); self.save_csv_button.pack(side=tk.LEFT, padx=5)
        self.save_image_button = ttk.Button(conv_control_frame_bottom, text="Save PNG...", command=self.save_output_to_png, state=tk.DISABLED); self.save_image_button.pack(side=tk.LEFT, padx=5)
        self.plot_3d_button = ttk.Button(conv_control_frame_bottom, text="Show 3D Plot", command=self.plot_output_in_3d, state=tk.DISABLED); self.plot_3d_button.pack(side=tk.LEFT, padx=5)
        compare_frame = ttk.Frame(conv_control_frame_bottom)
        compare_frame.pack(side=tk.LEFT, padx=5)
        self.compare_button = ttk.Button(compare_frame, text="Compare Modes...", command=self.run_comparison_workflow)
        self.compare_button.pack(side=tk.LEFT)
        self.show_plots_checkbox = ttk.Checkbutton(compare_frame, text="Show Plots", variable=self.show_plots_var)
        self.show_plots_checkbox.pack(side=tk.LEFT, padx=(5, 0))
        vis_controls_frame = ttk.Frame(conv_control_frame_bottom); vis_controls_frame.pack(side=tk.RIGHT, padx=10)
        ttk.Label(vis_controls_frame, text="Output Style:").pack(side=tk.LEFT)
        self.vis_mode_combo = ttk.Combobox(vis_controls_frame, values=["Grayscale", "Bipolar (RdBu)"], state="readonly", width=15); self.vis_mode_combo.pack(side=tk.LEFT)
        self.vis_mode_combo.bind("<<ComboboxSelected>>", lambda e: self.display_convolved_image())

    def on_power_selected(self, event=None): 
        self.master.title(f"Phototransistor Kernel Simulator ({self.power_combobox.get()})")
        self.update_all_kernel_cells_from_desired_weights()
    
    def update_all_kernel_cells_from_desired_weights(self):
        self.invalidate_convolution_results()
        if self.is_in_dynamic_mode():
            self.calculate_dynamic_regime()
        else:
            try:
                self.calculate_for_single_power(float(self.power_combobox.get()))
            except (ValueError, TypeError):
                self.update_step_text("Error: Invalid power level.")

    def handle_manual_weight_entry(self, event=None):
        self.current_preset_name = "CUSTOM"

    def apply_preset_kernel(self, kernel_name):
        preset = ALL_PRESET_KERNELS.get(kernel_name)
        if preset is None: return
        log.info(f"Applying preset kernel: {kernel_name}")
        self.current_preset_name = kernel_name
        for r_idx, c_idx in np.ndindex(3, 3):
            kernel_cells[r_idx, c_idx]['desired_w_entry'].delete(0, tk.END)
            kernel_cells[r_idx, c_idx]['desired_w_entry'].insert(0, f"{preset[r_idx, c_idx]:.4f}")
        self.update_all_kernel_cells_from_desired_weights()
    
    def calculate_for_single_power(self, power_level):
        global current_kernel_weights, current_kernel_voltages
        if not self.simulator: return

        self.update_step_text(f"Calculating for ND={power_level}... Please wait."); self.master.update_idletasks()

        try:
            unitless_kernel = np.array([[float(kernel_cells[r, c]['desired_w_entry'].get() or "0.0") for c in range(3)] for r in range(3)])
            
            result = self.simulator.run_fixed_power_calculation(
                unitless_kernel, power_level, self.nd5_voltage_cache, self.current_preset_name
            )

            current_kernel_voltages = result.voltages
            current_kernel_weights = result.currents

            # Update GUI labels
            for r_k, c_k in np.ndindex(3, 3):
                cell_info = kernel_cells[r_k, c_k]
                vtg = result.voltages[r_k, c_k]
                iph = result.currents[r_k, c_k]
                label_tag = "(Cached)" if power_level == 5.0 and self.nd5_voltage_cache else ""
                cell_info['found_vtg_label'].config(text=f"V_TG: {vtg:.2f}V {label_tag}")
                cell_info['actual_iph_label'].config(text=f"I_ph: {iph:.2e} A")
                cell_info['effective_nd_label'].config(text=f"Eff. ND: {power_level:.1f} (Fixed)")
            
            self.update_step_text(f"Kernel updated for ND={power_level}. Ready to convolve.")
            self.update_kernel_display(current_kernel_voltages)
            self.update_step_button_state()
            if input_image_np is not None: self.convolve_full_image_and_display()

        except (ValueError, RuntimeError) as e:
            self.update_step_text(f"Error: {e}")

    def calculate_dynamic_regime(self):
        global fixed_optimal_voltages, current_kernel_weights, current_kernel_voltages
        if not self.simulator: return

        self.update_step_text("Starting Dynamic Optimization..."); self.master.update_idletasks()

        try:
            unitless_kernel = np.array([[float(kernel_cells[r, c]['desired_w_entry'].get() or "0.0") for c in range(3)] for r in range(3)])

            # Pass the GUI's text update method as a callback
            result = self.simulator.run_dynamic_mode_optimization(
                unitless_kernel, progress_callback=self.update_step_text
            )

            fixed_optimal_voltages = result.voltages
            log.info(f"Optimizer found best voltages:\n{np.round(fixed_optimal_voltages, 2)}")

            # Update GUI state
            current_kernel_voltages = fixed_optimal_voltages.copy()
            current_kernel_weights.fill(0)

            self.update_step_text(f"Optimal V_tg found (derived from ND={result.ref_nd}, Loss={result.loss:.4f}). Ready.")

            for r_k, c_k in np.ndindex(3, 3):
                cell_info = kernel_cells[r_k, c_k]
                final_vtg = fixed_optimal_voltages[r_k, c_k]
                cell_info['found_vtg_label'].config(text=f"V_TG: {final_vtg:.2f}V (Fixed)")
                cell_info['effective_nd_label'].config(text="Eff. ND: (Dynamic)")
                cell_info['actual_iph_label'].config(text="I_ph: (Dynamic)")
            
            self.update_kernel_display(fixed_optimal_voltages)
            self.update_step_button_state()
            if input_image_np is not None: self.convolve_full_image_and_display()

        except (ValueError, RuntimeError) as e:
            self.update_step_text(f"Optimization failed: {e}")
    

    def convolve_full_image_and_display(self):
        global raw_convolved_output_amps, enable_debug_write_for_first_patch, normalized_grayscale_output

        # Reset the debug flag every time a full convolution starts
        enable_debug_write_for_first_patch = True
        
        # Clear the debug file for a fresh run
        if os.path.exists("DEBUG_LOG.txt"):
            os.remove("DEBUG_LOG.txt")

        if input_image_np is None: return
        normalized_input = normalize_input_image(input_image_np)
        raw_out = np.zeros_like(normalized_input, dtype=float)
        
        if self.is_in_dynamic_mode():
            # Dynamic Mode convolution still needs its special per-pixel logic
            H, W = normalized_input.shape
            padded = np.pad(normalized_input, ((1, 1), (1, 1)), mode="symmetric")
            for r in range(H):
                for c in range(W):
                    patch = padded[r:r + 3, c:c + 3]
                    adaptive_currents, _ = self.simulator.get_adaptive_kernel_weights(patch, fixed_optimal_voltages)
                    raw_out[r, c] = np.sum(adaptive_currents)
        else:
            # Fixed power mode can now use the simulator
            raw_out = self.simulator.run_convolution(normalized_input, current_kernel_weights)
        raw_convolved_output_amps = raw_out; 
        
                # Calculate and store the normalized grayscale data ---
        mn, mx = raw_out.min(), raw_out.max()
        if mx - mn > 1e-12:
            normalized_output_0_1 = (raw_out - mn) / (mx - mn)
        else:
            normalized_output_0_1 = np.zeros_like(raw_out)
        normalized_grayscale_output = (normalized_output_0_1 * 255).astype(np.uint8)
        
        self.display_convolved_image(); self.update_step_text(f"Full convolution complete.")
        # Enable saving and plotting buttons now that data is available
        self.save_csv_button.config(state=tk.NORMAL); self.plot_3d_button.config(state=tk.NORMAL)
        self.save_image_button.config(state=tk.NORMAL)
    
    def run_comparison_workflow(self):
        """
        Automatically compares Dynamic and Fixed Power modes. It always prints
        metrics and optionally shows plots based on the checkbox state.
        """
        if input_image_np is None or self.simulator is None:
            self.update_step_text("Error: Load an image first.")
            return

        try:
            ref_nd_level_str = self.power_combobox.get()
            if ref_nd_level_str == "Dynamic Mode":
                self.update_step_text("Please select a fixed ND level (e.g., 2.0) to serve as the reference.")
                return
            ref_nd_level = float(ref_nd_level_str)

            # ... (The calculation part is identical to before) ...
            self.update_step_text("Starting comparison workflow...")
            self.master.update_idletasks()
            unitless_kernel = np.array([[float(kernel_cells[r, c]['desired_w_entry'].get() or "0.0") for c in range(3)] for r in range(3)])
            normalized_input = normalize_input_image(input_image_np)
            self.update_step_text(f"Running reference convolution at ND={ref_nd_level}...")
            fixed_power_result = self.simulator.run_fixed_power_calculation(unitless_kernel, ref_nd_level, self.nd5_voltage_cache, self.current_preset_name)
            ref_raw_output = self.simulator.run_convolution(normalized_input, fixed_power_result.currents)
            self.update_step_text("Optimizing and running dynamic mode convolution...")
            dynamic_result = self.simulator.run_dynamic_mode_optimization(unitless_kernel)
            proc_raw_output = np.zeros_like(normalized_input, dtype=float)
            padded = np.pad(normalized_input, ((1, 1), (1, 1)), mode="symmetric")
            for r in range(normalized_input.shape[0]):
                for c in range(normalized_input.shape[1]):
                    patch = padded[r:r + 3, c:c + 3]
                    currents, _ = self.simulator.get_adaptive_kernel_weights(patch, dynamic_result.voltages)
                    proc_raw_output[r, c] = np.sum(currents)
            def to_grayscale(raw_array):
                mn, mx = raw_array.min(), raw_array.max()
                if mx - mn < 1e-12: return np.zeros_like(raw_array, dtype=np.uint8)
                norm = (raw_array - mn) / (mx - mn)
                return (norm * 255).astype(np.uint8)
            ref_grayscale = to_grayscale(ref_raw_output)
            proc_grayscale = to_grayscale(proc_raw_output)
            ref_name = f"Fixed_ND={ref_nd_level}"
            proc_name = "Dynamic_Mode"
            
            # --- NEW LOGIC STARTS HERE ---
            
            # 1. Always calculate and display metrics in the text box
            self.update_step_text("Calculating metrics...")
            metrics = image_comparator.calculate_metrics(ref_grayscale, proc_grayscale)
            metrics_display_text = f"--- Comparison: {self.current_preset_name} Kernel ---\n\n"
            for key, value in metrics.items():
                metrics_display_text += f"{key:<12}: {value:.4f}\n"
            
            # 2. Check the state of the checkbox
            if self.show_plots_var.get():
                metrics_display_text += "\nGenerating plot..."
                self.update_step_text(metrics_display_text)
                
                # Generate and display the plot in a new window
                fig_imgs = self._create_comparison_figure(ref_grayscale, proc_grayscale, ref_name, proc_name)
                PlotWindow(self.master, fig_imgs, title="Image Comparison")
            else:
                # If not showing plots, just show the final metrics
                self.update_step_text(metrics_display_text)

        except Exception as e:
            log.error(f"Comparison workflow failed: {e}", exc_info=True)
            self.update_step_text(f"Error during comparison: {e}")

    def _create_comparison_figure(self, ref_img, proc_img, ref_name, proc_name):
        """Helper method that creates and returns the visual comparison figure."""
        fig_imgs, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig_imgs.suptitle("Image Comparison: Visual Analysis", fontsize=16)

        # Plot Reference Image
        axes[0].imshow(ref_img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title(f"Reference Image\n('{ref_name}')")
        axes[0].axis('off')

        # Plot Processed Image
        axes[1].imshow(proc_img, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f"Processed Image\n('{proc_name}')")
        axes[1].axis('off')

        # Plot Difference Map
        diff_img = ref_img.astype(np.int16) - proc_img.astype(np.int16)
        vabs_max = np.max(np.abs(diff_img))
        im = axes[2].imshow(diff_img, cmap='RdBu_r', vmin=-vabs_max, vmax=vabs_max)
        axes[2].set_title("Difference Map")
        axes[2].axis('off')

        # Robust color bar placement
        fig_imgs.tight_layout(rect=[0, 0, 0.9, 0.95])
        cax = fig_imgs.add_axes([0.92, 0.15, 0.02, 0.7])
        fig_imgs.colorbar(im, cax=cax)
        
        return fig_imgs

    def perform_one_convolution_step(self):
        global current_conv_r, current_conv_c, convolution_active, output_image_step_by_step_amps
        if not convolution_active: return
        img_height, img_width = input_image_np.shape
        
        # Check if we are done stepping
        if current_conv_r >= img_height:
            convolution_active = False; self.update_step_button_state()
            global raw_convolved_output_amps; raw_convolved_output_amps = output_image_step_by_step_amps.copy()
            self.update_step_text("Convolution complete."); 
            self.save_image_button.config(state=tk.NORMAL) 
            self.save_csv_button.config(state=tk.NORMAL); self.plot_3d_button.config(state=tk.NORMAL)
            return
        stimulus_patch = padded_input_image_for_step_normalized[current_conv_r:current_conv_r+3, current_conv_c:current_conv_c+3]
        if self.is_in_dynamic_mode():
            achieved_currents, effective_nds = self.simulator.get_adaptive_kernel_weights(stimulus_patch, fixed_optimal_voltages)
            raw_pixel_value = np.sum(achieved_currents)
            weights_for_display = achieved_currents
            
            # Update GUI labels for dynamic mode
            for r_k, c_k in np.ndindex(3, 3): 
                kernel_cells[r_k, c_k]['effective_nd_label'].config(text=f"Eff. ND: {effective_nds[r_k, c_k]:.2f}")
                kernel_cells[r_k, c_k]['actual_iph_label'].config(text=f"I_ph: {weights_for_display[r_k, c_k]:.2e} A")
        else: # Fixed Power mode
            weights_for_display = current_kernel_weights
            raw_pixel_value = np.sum(stimulus_patch * weights_for_display)
        output_image_step_by_step_amps[current_conv_r, current_conv_c] = raw_pixel_value
        self.display_convolved_image(step_by_step=True)
        self.update_convolution_step_display(current_conv_r, current_conv_c, stimulus_patch, weights_for_display, raw_pixel_value)
        
        # Advance to the next pixel
        current_conv_c += 1
        if current_conv_c >= img_width: current_conv_c, current_conv_r = 0, current_conv_r + 1
        if current_conv_r >= img_height: convolution_active = False
        self.update_step_button_state()


    # --- GUI Update and Display Methods ---
    
    def update_kernel_display(self, voltages_to_display):
        self.kernel_ax.clear()
        if voltages_to_display is not None:
            abs_max = np.max(np.abs(voltages_to_display)) if voltages_to_display.size > 0 else 1.0
            if abs_max < 1e-9: abs_max = 1.0
            self.kernel_ax.matshow(voltages_to_display, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
            for i, j in np.ndindex(3, 3): self.kernel_ax.text(j, i, f"{voltages_to_display[i, j]:.2f} V", ha="center", va="center", color="black", fontsize=9)
        self.kernel_ax.set_xticks([]); self.kernel_ax.set_yticks([])
        self.kernel_canvas_widget.draw()

    def plot_dynamic_response(self):
        if not self.is_in_dynamic_mode() or np.all(fixed_optimal_voltages == 0):
            self.update_step_text("Plotting requires running the optimizer in 'Dynamic Mode' first."); return
        plot_window = tk.Toplevel(self.master); plot_window.title("Pixel Dynamic Response Curves (at Optimal Voltages)"); plot_window.geometry("1000x850") 
        TITLE_FONT_SIZE = 16; SUBPLOT_TITLE_FONT_SIZE = 10; AXIS_LABEL_FONT_SIZE = 12; LEGEND_FONT_SIZE = 10
        fig, axes = plt.subplots(3, 3, figsize=(10, 8.5), sharex=True, sharey=True); fig.suptitle("Achieved Current vs. Optical Power for each Pixel", fontsize=TITLE_FONT_SIZE)
        min_x = min(k[0] for k in self.full_char_data.keys()); min_y = min(k[1] for k in self.full_char_data.keys())
        
        # Auto-scale the y-axis for better readability
        center_pixel_coord = (min_x + 1, min_y + 1); center_char = self.full_char_data.get(center_pixel_coord)
        y_scale_factor = 1.0
        if center_char:
            max_abs_current = np.max(np.abs(center_char.i_ph_vs_nd_original.values))
            if max_abs_current > 1e-12: y_scale_factor = 10**np.floor(np.log10(max_abs_current))
        exponent = int(np.log10(y_scale_factor)) if y_scale_factor > 0 else 0
        for r_k, c_k in np.ndindex(3, 3):
            ax = axes[r_k, c_k]; pixel_coord = (min_x + c_k, min_y + r_k); vtg = fixed_optimal_voltages[r_k, c_k]
            char = self.full_char_data.get(pixel_coord)
            if not char:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes); ax.set_xticks([]); ax.set_yticks([]); continue
            measured_nd = char.nd_axis
            measured_iph = np.array([get_iph_for_vtg_interpolated(vtg, nd, char) for nd in measured_nd]) / y_scale_factor
            ax.plot(measured_nd, measured_iph, 'o', label='Measured NDs', color='red', zorder=10, markersize=5)
            interp_nd_range = np.linspace(char.nd_axis.min(), char.nd_axis.max(), 100)
            interp_iph = np.array([get_iph_for_vtg_interpolated(vtg, nd, char) for nd in interp_nd_range]) / y_scale_factor
            ax.plot(interp_nd_range, interp_iph, '-', label='Interpolated Response', zorder=5, linewidth=2)
            ax.set_title(f"Pixel ({pixel_coord[0]},{pixel_coord[1]}), V_TG = {vtg:.2f}V", fontsize=SUBPLOT_TITLE_FONT_SIZE, pad=3)
            ax.grid(True, linestyle=':', linewidth=0.5)
        handles, labels = axes[0,0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.98, 0.08), fontsize=LEGEND_FONT_SIZE)
        fig.supxlabel("Optical Power (ND Filter Value)", fontsize=AXIS_LABEL_FONT_SIZE); fig.supylabel(f"Achieved Photocurrent ($\\times 10^{{{exponent}}}$ A)", fontsize=AXIS_LABEL_FONT_SIZE)
        axes[0,0].yaxis.set_major_formatter(plt.matplotlib.ticker.ScalarFormatter(useMathText=True)); axes[0,0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        fig.tight_layout(rect=[0.03, 0.03, 1, 0.95])
        canvas = FigureCanvasTkAgg(fig, master=plot_window); canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_step_button_state(self):
        is_ready = (fixed_optimal_voltages is not None and not np.all(fixed_optimal_voltages==0)) if self.is_in_dynamic_mode() else (current_kernel_weights is not None and not np.all(current_kernel_weights==0))
        is_ready = is_ready and (input_image_np is not None)
        if hasattr(self, 'plot_response_button'):
            dynamic_ready = self.is_in_dynamic_mode() and is_ready
            self.plot_response_button.config(state=tk.NORMAL if dynamic_ready else tk.DISABLED)
        self.reset_step_button.config(state=tk.NORMAL if is_ready else tk.DISABLED)
        self.convolve_full_button.config(state=tk.NORMAL if is_ready else tk.DISABLED)
        self.compare_button.config(state=tk.NORMAL if is_ready else tk.DISABLED)
        self.show_plots_checkbox.config(state=tk.NORMAL if is_ready else tk.DISABLED)
        img_height = input_image_np.shape[0] if input_image_np is not None else 0
        is_stepping = convolution_active and current_conv_r < img_height
        self.next_step_button.config(state=tk.NORMAL if is_stepping else tk.DISABLED); self.five_steps_button.config(state=tk.NORMAL if is_stepping else tk.DISABLED); self.finish_steps_button.config(state=tk.NORMAL if is_stepping else tk.DISABLED)
        if not is_ready: self.update_step_text("Load an image." if input_image_np is None else "Update/Optimize kernel.")
        elif not is_stepping and convolution_active: self.update_step_text("Convolution complete.")

    def start_reset_step_by_step(self):
        global convolution_active
        is_ready = (fixed_optimal_voltages is not None and not np.all(fixed_optimal_voltages==0)) if self.is_in_dynamic_mode() else (current_kernel_weights is not None and not np.all(current_kernel_weights==0))
        if input_image_np is None: self.update_step_text("Load image first."); return
        if not is_ready: self.update_step_text("Update/Optimize kernel first."); return
        log.info("Resetting convolution state for stepping."); self.invalidate_convolution_results(); self.reset_convolution_state()
        if padded_input_image_for_step_normalized is None: self.update_step_text("Error: Could not prepare image data."); convolution_active = False
        else: convolution_active = True; self.update_step_text("Step-by-step ready. Click 'Next Step'.")
        self.update_step_button_state()

    def reset_convolution_state(self):
        global current_conv_r, current_conv_c, padded_input_image_for_step_normalized, output_image_step_by_step_amps
        current_conv_r, current_conv_c = 0, 0
        if input_image_np is not None:
            normalized_input = normalize_input_image(input_image_np)
            padded_input_image_for_step_normalized = np.pad(normalized_input, ((1, 1), (1, 1)), mode="symmetric")
            output_image_step_by_step_amps = np.zeros_like(normalized_input)
            self.display_convolved_image(step_by_step=True)
        else:
            padded_input_image_for_step_normalized = output_image_step_by_step_amps = None; self.display_convolved_image()

    # --- Image Loading ---
    def load_image(self):
        global input_image_np; filepath = filedialog.askopenfilename(filetypes=(("PNG", "*.png"),("All", "*.*")));
        if not filepath: return
        img_pil = Image.open(filepath).convert('L'); input_image_np = np.array(img_pil)
        self._display_numpy_image_on_canvas(input_image_np, self.input_ax, self.input_canvas)
        log.info(f"Image loaded: {filepath}"); self.invalidate_convolution_results(); self.reset_convolution_state(); 

    def _load_numpy_image(self, image_array, description):
        global input_image_np; input_image_np = image_array.copy()
        self._display_numpy_image_on_canvas(input_image_np, self.input_ax, self.input_canvas)
        log.info(f"{description} loaded"); self.invalidate_convolution_results(); self.reset_convolution_state()
        if self.full_char_data: self.update_step_button_state()

    def _display_numpy_image_on_canvas(self, np_image, target_ax, target_canvas):
        """Displays a NumPy array on a specified Matplotlib canvas."""
        target_ax.clear()
        if np_image is not None:
            target_ax.imshow(np_image, cmap='gray', vmin=0, vmax=255)
        target_ax.set_xticks([])
        target_ax.set_yticks([])
        target_canvas.draw()

    def load_default_vertical_line(self): self._load_default_image("vert")
    def load_default_horizontal_line(self): self._load_default_image("horiz")
    def load_default_center_dot(self): self._load_default_image("dot")
    def load_default_eye(self):
        pattern = np.array([[0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0],[0,1,0,1,1,1,0,0],[1,0,0,1,1,0,1,0],[0,1,0,0,0,1,0,0],[0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]])
        self._load_numpy_image((pattern * 255).astype(np.uint8), "Default 'Eye' Icon")
    def load_default_blitz(self):
        pattern = np.array([[0,0,0,0,1,1,1,1],[0,0,0,1,1,1,1,0],[0,0,1,1,1,0,0,0],[0,1,1,1,1,1,1,0],[0,0,0,1,1,1,0,0],[0,0,1,1,1,0,0,0],[0,1,1,0,0,0,0,0],[0,1,0,0,0,0,0,0]])
        self._load_numpy_image((pattern * 255).astype(np.uint8), "Default 'Blitz' Icon")
    def load_default_sword(self):
        pattern = np.array([[0,0,0,0,0,0,1,1],[0,0,0,0,0,1,0,1],[0,0,0,0,1,0,1,0],[1,0,0,1,0,1,0,0],[0,1,1,0,1,0,0,0],[0,0,1,1,0,0,0,0],[0,1,0,1,0,0,0,0],[1,0,0,0,1,0,0,0]])
        self._load_numpy_image((pattern * 255).astype(np.uint8), "Default 'Sword' Icon")
    def _load_default_image(self, shape_type):
        img = np.zeros((8, 8), dtype=np.uint8)
        if shape_type == "vert": img[:, 3:5] = 255
        elif shape_type == "horiz": img[3:5, :] = 255
        elif shape_type == "dot": img[3:5, 3:5] = 255
        self._load_numpy_image(img, f"Default {shape_type}")
    def _load_zx_spectrum_char(self, char):
        pattern = ZX_SPECTRUM_FONT.get(char.upper())
        if pattern is not None: self._load_numpy_image((pattern * 255).astype(np.uint8), f"ZX Letter '{char}'")

    def invalidate_convolution_results(self):
        global raw_convolved_output_amps,normalized_grayscale_output
        raw_convolved_output_amps = None
        normalized_grayscale_output = None
        self.display_convolved_image(); self.save_csv_button.config(state=tk.DISABLED); 
        self.save_image_button.config(state=tk.DISABLED) 
        self.plot_3d_button.config(state=tk.DISABLED)

    def save_output_to_csv(self):
       if raw_convolved_output_amps is None or normalized_grayscale_output is None:
           self.update_step_text("Error: No convolution data available to save.")
           return
       
       # --- NEW: Use the custom dialog ---
       dialog = SaveOptionsDialog(self.master)
       save_format = dialog.result # This will be 'raw', 'grayscale', or None

       if save_format is None:
           log.info("Save operation cancelled by user.")
           return # User closed or cancelled the dialog

       # Ask for the file path *after* they have chosen the format
       filepath = filedialog.asksaveasfilename(
           defaultextension=".csv",
           filetypes=[("CSV Files", "*.csv")]
       )
       if not filepath:
           return # User cancelled the file dialog

       # Select the data and format based on the chosen format
       if save_format == "raw":
           data_to_save = raw_convolved_output_amps
           fmt = '%.6e' # Use scientific notation for currents
           log.info(f"Saving raw current data (Amps) to {filepath}")
       else: # save_format == "grayscale"
           data_to_save = normalized_grayscale_output
           fmt = '%d' # Use integer format for grayscale
           log.info(f"Saving normalized grayscale data (0-255) to {filepath}")

       # Save the chosen data to the CSV file
       np.savetxt(filepath, data_to_save, delimiter=',', fmt=fmt)
       self.update_step_text(f"Successfully saved output to {filepath}")
    
    def save_output_to_png(self):
        """Saves the current output display as a PNG image."""
        if raw_convolved_output_amps is None:
            self.update_step_text("Error: No convolution data available to save.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")]
        )
        if not filepath:
            log.info("Save image operation cancelled.")
            return

        try:
            # Save the entire figure, cropping tightly to the image area.
            # This is the most reliable way to get what the user sees.
            self.output_fig.savefig(
                filepath, 
                dpi=300, 
                bbox_inches='tight', 
                pad_inches=0.1
            )
            log.info(f"Successfully saved output image to {filepath}")
            self.update_step_text(f"Successfully saved output image to {filepath}")
        except Exception as e:
            log.error(f"Failed to save image: {e}")
            self.update_step_text(f"Error saving image: {e}")

    def plot_output_in_3d(self):
       if raw_convolved_output_amps is None: return
       plot_window = tk.Toplevel(self.master); plot_window.title("3D Output Current Heatmap")
       fig = plt.Figure(figsize=(8, 7), dpi=100); ax = fig.add_subplot(111, projection='3d')
       rows, cols = raw_convolved_output_amps.shape; x, y = np.meshgrid(np.arange(cols), np.arange(rows)); z = raw_convolved_output_amps.flatten()
       vabs_max = np.max(np.abs(z)); norm = plt.Normalize(-vabs_max if vabs_max > 0 else -1, vabs_max if vabs_max > 0 else 1); colors = plt.get_cmap('RdBu_r')(norm(z))
       ax.bar3d(x.flatten(), y.flatten(), np.zeros(rows*cols), 0.8, 0.8, z, color=colors)
       ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Iph (A)"); ax.invert_yaxis()
       sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm); sm.set_array([]); fig.colorbar(sm, ax=ax, shrink=0.6, aspect=10, label="Current (A)")
       canvas = FigureCanvasTkAgg(fig, master=plot_window); canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_step_text(self, message): 
        if hasattr(self, 'convolution_step_text') and self.convolution_step_text.winfo_exists():
            self.convolution_step_text.config(state=tk.NORMAL); self.convolution_step_text.delete(1.0, tk.END); self.convolution_step_text.insert(tk.END, message); self.convolution_step_text.config(state=tk.DISABLED)
            if self.master.winfo_exists(): self.master.update_idletasks()

    def update_convolution_step_display(self, r, c, stimulus_patch, kernel_patch_amps, output_pixel_amps):
        fmt = {"float_kind": lambda x: f"{x:.3e}"}
        if self.is_in_dynamic_mode():
            text = (f"Kernel Center @ ({r},{c})\n\n" f"Norm. Stimulus Patch:\n{np.array2string(stimulus_patch, precision=2)}\n\n" f"Achieved Currents [A]:\n{np.array2string(kernel_patch_amps, formatter=fmt)}\n\n" f"Output (Sum of Currents) [A]: {output_pixel_amps:.3e}")
        else:
            element_products = stimulus_patch * kernel_patch_amps
            text = (f"Kernel Center @ ({r},{c})\n\n" f"Norm. Stimulus Patch:\n{np.array2string(stimulus_patch, precision=2)}\n\n" f"Kernel Weights [A]:\n{np.array2string(kernel_patch_amps, formatter=fmt)}\n\n" f"Element Products [A]:\n{np.array2string(element_products, formatter=fmt)}\n\n" f"Output (Sum of Products) [A]: {output_pixel_amps:.3e}")
        self.update_step_text(text)
        
    def display_convolved_image(self, step_by_step=False):
        data = output_image_step_by_step_amps if step_by_step else raw_convolved_output_amps
        if hasattr(self, 'colorbar'): self.colorbar.remove(); delattr(self, 'colorbar')
        self.output_ax.clear()
        if data is None or data.size == 0:
            self.output_ax.set_xticks([]); self.output_ax.set_yticks([]); self.output_canvas.draw(); return
        vis_mode = self.vis_mode_combo.get()
        cmap = 'gray' if vis_mode == "Grayscale" else 'RdBu_r'
        if cmap == 'RdBu_r': vabs_max = np.max(np.abs(data)); vmin, vmax = -vabs_max if vabs_max > 1e-12 else -1, vabs_max if vabs_max > 1e-12 else 1
        else: vmin, vmax = data.min(), data.max()
        if abs(vmax - vmin) < 1e-12: vmin, vmax = data.min() - 0.5, data.max() + 0.5
        im = self.output_ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        self.colorbar = self.output_fig.colorbar(im, ax=self.output_ax, fraction=0.046, pad=0.04)
        self.colorbar.set_label("Photocurrent (A)")
        self.output_ax.set_xticks([]); self.output_ax.set_yticks([]); self.output_fig.tight_layout(pad=0.5); self.output_canvas.draw()
        
    def perform_n_convolution_steps(self, num_steps):
        for _ in range(num_steps):
            if not convolution_active: break
            self.perform_one_convolution_step()

    def finish_all_convolution_steps(self):
        self.update_step_text("Finishing remaining steps... Please wait.")
        while convolution_active: self.perform_one_convolution_step(); self.master.update_idletasks()

def launch_gui() -> None:
    root = tk.Tk()
    ConvolutionApp(root)
    root.mainloop()