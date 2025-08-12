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
            if hasattr(self, 'pixel_x_entry'):
                self.pixel_x_entry.config(state=tk.DISABLED)
                self.pixel_y_entry.config(state=tk.DISABLED)
            if hasattr(self, 'plot_char_surface_button'):
                self.plot_char_surface_button.config(state=tk.DISABLED)
            if hasattr(self, 'plot_ratio_button'):
                self.plot_ratio_button.config(state=tk.DISABLED)
            if hasattr(self, 'plot_response_button'):
                self.plot_response_button.config(state=tk.DISABLED)

            
    def is_in_dynamic_mode(self): return self.power_combobox.get() == "Dynamic Mode"

    def setup_main_layout(self):
        # Main scrollable canvas setup
        self.canvas = tk.Canvas(self.master); self.scrollbar_y = ttk.Scrollbar(self.master, orient="vertical", command=self.canvas.yview); self.scrollbar_x = ttk.Scrollbar(self.master, orient="horizontal", command=self.canvas.xview); self.scrollable_frame = ttk.Frame(self.canvas); self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))); self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw"); self.canvas.configure(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set); self.canvas.grid(row=0, column=0, sticky="nsew"); self.scrollbar_y.grid(row=0, column=1, sticky="ns"); self.scrollbar_x.grid(row=1, column=0, sticky="ew"); self.master.grid_rowconfigure(0, weight=1); self.master.grid_columnconfigure(0, weight=1)
        self.content_main_frame = ttk.Frame(self.scrollable_frame, padding="10"); self.content_main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Define all major frames first ---
        self.row0_frame = ttk.Frame(self.content_main_frame); self.row0_frame.pack(fill=tk.X, pady=5) 
        self.row1_frame = ttk.Frame(self.content_main_frame); self.row1_frame.pack(fill=tk.X, pady=10)
        self.row2_frame = ttk.Frame(self.content_main_frame); self.row2_frame.pack(fill=tk.X, pady=5)
        self.bottom_row_frame = ttk.Frame(self.content_main_frame); self.bottom_row_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # --- Row 0: Setup Controls (Images, Device, Kernels) ---
        self.image_controls_frame = ttk.LabelFrame(self.row0_frame, text="Default Images"); self.image_controls_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.device_control_frame = ttk.LabelFrame(self.row0_frame, text="Device Control"); self.device_control_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.presets_frame = ttk.LabelFrame(self.row0_frame, text="Preset Kernels (Unitless)"); self.presets_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Populate Image Controls
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

        # Populate Device Control
        ttk.Label(self.device_control_frame, text="Optical Power (ND):").pack(pady=(5,2))
        self.power_combobox = ttk.Combobox(self.device_control_frame, state="readonly", width=12)
        self.power_combobox.pack(pady=(0,5), padx=5)
        self.power_combobox.bind("<<ComboboxSelected>>", self.on_power_selected)
        gamma_val = 2.2 if cfg.dynamic_mode_use_gamma_correction else 1.0
        ttk.Label(self.device_control_frame, text=f"Dyn. Gamma: {gamma_val:.1f}").pack(pady=(0, 5))

        # Populate Preset Kernels
        for idx, kernel_name in enumerate(ALL_PRESET_KERNELS.keys()):
            btn = ttk.Button(self.presets_frame, text=kernel_name, width=12, command=lambda kn=kernel_name: self.apply_preset_kernel(kn))
            btn.grid(row=idx // 3, column=idx % 3, padx=1, pady=1, sticky="ew")

        # --- Row 1: Kernel Config and Plotting Tools ---
        self.kernel_config_frame = ttk.LabelFrame(self.row1_frame, text="Kernel Cell Config"); self.kernel_config_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y, anchor='n')
        self.plotting_tools_frame = ttk.LabelFrame(self.row1_frame, text="Plotting Tools"); self.plotting_tools_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y, anchor='n')
        self.kernel_display_frame = ttk.Frame(self.row1_frame); self.kernel_display_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True, anchor='n')

        # Populate Kernel Config
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
        
        # Populate Plotting Tools
        pixel_select_frame = ttk.Frame(self.plotting_tools_frame)
        pixel_select_frame.pack(pady=5, padx=5)
        ttk.Label(pixel_select_frame, text="Pixel X:").pack(side=tk.LEFT)
        self.pixel_x_entry = ttk.Entry(pixel_select_frame, width=4)
        self.pixel_x_entry.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(pixel_select_frame, text="Y:").pack(side=tk.LEFT)
        self.pixel_y_entry = ttk.Entry(pixel_select_frame, width=4)
        self.pixel_y_entry.pack(side=tk.LEFT)
        self.plot_response_button = ttk.Button(self.plotting_tools_frame, text="Plot Dynamic Response", command=self.plot_dynamic_response, state=tk.DISABLED)
        self.plot_response_button.pack(pady=5, padx=5, fill=tk.X)
        self.plot_char_surface_button = ttk.Button(self.plotting_tools_frame, text="Plot I-V-ND Surface", command=self.plot_characterization_surface)
        self.plot_char_surface_button.pack(pady=5, padx=5, fill=tk.X)
        self.plot_ratio_button = ttk.Button(self.plotting_tools_frame, text="Plot Stability Ratios", command=self.plot_stability_ratios)
        self.plot_ratio_button.pack(pady=5, padx=5, fill=tk.X)
        ratio_range_frame = ttk.Frame(self.plotting_tools_frame)
        ratio_range_frame.pack(pady=5, padx=5, fill=tk.X)
        ttk.Label(ratio_range_frame, text="V_tg Range:").pack(side=tk.LEFT)
        self.vtg_min_entry = ttk.Entry(ratio_range_frame, width=5)
        self.vtg_min_entry.insert(0, "1.0")
        self.vtg_min_entry.pack(side=tk.LEFT, padx=(2,0))
        ttk.Label(ratio_range_frame, text="to").pack(side=tk.LEFT, padx=2)
        self.vtg_max_entry = ttk.Entry(ratio_range_frame, width=5)
        self.vtg_max_entry.insert(0, "4.0")
        self.vtg_max_entry.pack(side=tk.LEFT, padx=(0,2))

        # Populate Kernel Display
        ttk.Label(self.kernel_display_frame, text="Kernel Gate Voltages (V_TG) [V]").pack()
        self.kernel_fig, self.kernel_ax = plt.subplots(figsize=(2.2, 2.2)); self.kernel_canvas_widget = FigureCanvasTkAgg(self.kernel_fig, master=self.kernel_display_frame); self.kernel_canvas_widget.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Row 2: Action Controls (Convolution, Saving) ---
        self.conv_control_outer_frame = ttk.LabelFrame(self.row2_frame, text="Convolution Control"); self.conv_control_outer_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        self.saving_frame = ttk.LabelFrame(self.row2_frame, text="File Saving"); self.saving_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.Y)
        
        # Populate Convolution Control
        conv_control_frame_top = ttk.Frame(self.conv_control_outer_frame); conv_control_frame_top.pack(fill=tk.X, expand=True, pady=(5, 2), padx=5)
        conv_control_frame_bottom = ttk.Frame(self.conv_control_outer_frame); conv_control_frame_bottom.pack(fill=tk.X, expand=True, pady=(2, 5), padx=5)
        self.convolve_full_button = ttk.Button(conv_control_frame_top, text="Convolve Full Image", command=self.convolve_full_image_and_display); self.convolve_full_button.pack(side=tk.LEFT, padx=5)
        self.reset_step_button = ttk.Button(conv_control_frame_top, text="Start/Reset Step-by-Step", command=self.start_reset_step_by_step); self.reset_step_button.pack(side=tk.LEFT, padx=5)
        self.next_step_button = ttk.Button(conv_control_frame_top, text="Next Step", command=self.perform_one_convolution_step, state=tk.DISABLED); self.next_step_button.pack(side=tk.LEFT, padx=5)
        self.five_steps_button = ttk.Button(conv_control_frame_top, text="5 Steps", command=lambda: self.perform_n_convolution_steps(5), state=tk.DISABLED); self.five_steps_button.pack(side=tk.LEFT, padx=5)
        self.finish_steps_button = ttk.Button(conv_control_frame_top, text="Finish Steps", command=self.finish_all_convolution_steps, state=tk.DISABLED); self.finish_steps_button.pack(side=tk.LEFT, padx=5)
        self.plot_3d_button = ttk.Button(conv_control_frame_bottom, text="Show 3D Plot", command=self.plot_output_in_3d, state=tk.DISABLED); self.plot_3d_button.pack(side=tk.LEFT, padx=5)
        compare_frame = ttk.Frame(conv_control_frame_bottom)
        compare_frame.pack(side=tk.LEFT, padx=5)
        self.compare_button = ttk.Button(compare_frame, text="Compare Modes...", command=self.run_comparison_workflow)
        self.compare_button.pack(side=tk.LEFT)
        self.show_plots_checkbox = ttk.Checkbutton(compare_frame, text="Show Plots", variable=self.show_plots_var)
        self.show_plots_checkbox.pack(side=tk.LEFT, padx=(5, 0))
        vis_controls_frame = ttk.Frame(conv_control_frame_bottom); vis_controls_frame.pack(side=tk.RIGHT, padx=10)
        ttk.Label(vis_controls_frame, text="Output Style:").pack(side=tk.LEFT)
        self.vis_mode_combo = ttk.Combobox(vis_controls_frame, values=["Grayscale", "Bipolar (RdBu)", "Zero Crossings"], state="readonly", width=15); self.vis_mode_combo.pack(side=tk.LEFT)
        self.vis_mode_combo.bind("<<ComboboxSelected>>", lambda e: self.display_convolved_image())

        # Populate Saving Frame
        self.save_csv_button = ttk.Button(self.saving_frame, text="Save CSV...", command=self.save_output_to_csv, state=tk.DISABLED)
        self.save_csv_button.pack(pady=5, padx=5, fill=tk.X)
        self.save_image_button = ttk.Button(self.saving_frame, text="Save PNG...", command=self.save_output_to_png, state=tk.DISABLED)
        self.save_image_button.pack(pady=5, padx=5, fill=tk.X)

        # --- Bottom Row: Display Panels ---
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

        # Pack Bottom Row
        self.input_display_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.input_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.output_display_frame.pack(side=tk.LEFT, padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.output_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.animation_frame.pack(side=tk.RIGHT, padx=10, pady=5, fill=tk.Y, expand=False)
        self.convolution_step_text.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)

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

    def _create_zero_crossing_image(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Creates a binary image highlighting zero-crossings in the raw data.
        A crossing is marked if adjacent pixels have different signs.
        This correctly handles transitions to/from zero.
        """
        if raw_data is None or raw_data.size == 0:
            return np.zeros((1, 1), dtype=np.uint8)

        # Use a small tolerance for floating point comparisons
        tolerance = 1e-12

        # A pixel's "sign" is determined by whether it's positive or not.
        # This means zero and negative values are grouped together.
        is_positive = raw_data > tolerance

        # A crossing occurs if the 'is_positive' status flips.
        # This correctly handles [pos, neg], [pos, zero], [neg, pos], and [zero, pos].
        horizontal_crossings_pos = np.logical_xor(is_positive[:, :-1], is_positive[:, 1:])
        vertical_crossings_pos = np.logical_xor(is_positive[:-1, :], is_positive[1:, :])

        # To catch the missing cases ([neg, zero] and [zero, neg]), we do the same
        # check for whether a pixel is negative or not.
        is_negative = raw_data < -tolerance
        horizontal_crossings_neg = np.logical_xor(is_negative[:, :-1], is_negative[:, 1:])
        vertical_crossings_neg = np.logical_xor(is_negative[:-1, :], is_negative[1:, :])

        # A final crossing map is the union (OR) of both types of crossings.
        final_horizontal = np.logical_or(horizontal_crossings_pos, horizontal_crossings_neg)
        final_vertical = np.logical_or(vertical_crossings_pos, vertical_crossings_neg)

        # Initialize the output image and apply the crossings
        zero_crossings_mask = np.zeros_like(raw_data, dtype=bool)
        zero_crossings_mask[:, :-1] |= final_horizontal
        zero_crossings_mask[:-1, :] |= final_vertical

        # Convert the boolean mask to a displayable image (0 for black, 255 for white)
        return (zero_crossings_mask * 255).astype(np.uint8)


    # --- GUI Update and Display Methods ---
    
    def update_kernel_display(self, voltages_to_display):
        self.kernel_ax.clear()

        # Revert the figure background to its default for a clean look
        default_bg = plt.rcParams['figure.facecolor']
        self.kernel_fig.set_facecolor(default_bg)
        self.kernel_ax.set_facecolor(default_bg)

        if voltages_to_display is not None:
            abs_max = np.max(np.abs(voltages_to_display)) if voltages_to_display.size > 0 else 1.0
            if abs_max < 1e-9: abs_max = 1.0
            
            # 1. Define the colormap and the normalization range
            cmap = plt.get_cmap('RdBu_r')  # Using the Red-Blue colormap as requested
            norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)

            # 2. Draw the background matrix using the first line you provided
            self.kernel_ax.matshow(voltages_to_display, cmap=cmap, norm=norm)
            
            # 3. Loop through each cell to set the adaptive text color
            for i, j in np.ndindex(3, 3):
                value = voltages_to_display[i, j]
                
                # Get the RGBA background color of the cell from the colormap
                bg_color = cmap(norm(value))
                
                # Calculate the perceived brightness (luminance) of the background
                # This is a standard formula for converting RGB to brightness
                luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
                
                # If the background is dark (luminance < 0.5), use white text.
                # If the background is light, use black text.
                text_color = 'white' if luminance < 0.5 else 'black'
                
                # This is the second line you provided, now with the adaptive `text_color`
                self.kernel_ax.text(j, i, f"{value:.2f} V",
                                    ha="center", va="center", color=text_color, fontsize=9)

        self.kernel_ax.set_xticks([])
        self.kernel_ax.set_yticks([])
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
    
    def _get_selected_pixel_char(self):
        """
        Reads pixel coordinates from GUI, validates them, and returns the
        corresponding characterization data. Handles all error messages.
        Returns a tuple (char_data, pixel_coord) or None on failure.
        """
        if not self.full_char_data:
            self.update_step_text("Error: Characterization data not loaded.")
            return None
        try:
            pixel_x = int(self.pixel_x_entry.get())
            pixel_y = int(self.pixel_y_entry.get())
        except ValueError:
            self.update_step_text("Error: Invalid pixel coordinates. Please enter integers.")
            return None

        pixel_coord = (pixel_x, pixel_y)
        char_data = self.full_char_data.get(pixel_coord)

        if char_data is None:
            # Provide helpful info on available pixels
            available_keys = list(self.full_char_data.keys())
            self.update_step_text(f"Error: No data for pixel {pixel_coord}.\n"
                                  f"Available pixels are: {available_keys[:5]}...")
            return None
            
        return char_data, pixel_coord

    def plot_characterization_surface(self):
        """
        Generates a 3D surface plot for the USER-SELECTED pixel using a
        manual symmetric natural logarithmic transformation.
        """        
        # Use the helper function to get the selected pixel's data
        result = self._get_selected_pixel_char()
        if result is None:
            return  # Error message was already shown by the helper
        center_char, center_pixel_coord = result

        # Filter out unwanted ND levels before plotting
        i_ph_df = center_char.i_ph_vs_nd.copy()
        if 6.0 in i_ph_df.columns:
            i_ph_df = i_ph_df.drop(columns=[6.0])
            log.info("Filtered out ND=6.0 from the 3D surface plot.")
        
        v_tg_axis = i_ph_df.index.to_numpy()
        i_ph_data = i_ph_df.to_numpy()

        # Create meshgrid for surface plot
        nd_axis_plot = i_ph_df.columns.to_numpy(dtype=float)
        ND, V_TG = np.meshgrid(nd_axis_plot, v_tg_axis)
        I_PH = i_ph_data

        # Manual Symmetric Log Transformation of Z-data
        linthresh = 1e-9 
        I_PH_TRANSFORMED = np.sign(I_PH) * np.log1p(np.abs(I_PH) / linthresh)

        # Valley Trace Calculation
        nd_valley_points, vtg_valley_points, iph_valley_points = [], [], []
        for nd_level in i_ph_df.columns:
            currents_for_this_nd = i_ph_df[nd_level].to_numpy()
            min_abs_current_idx = np.argmin(np.abs(currents_for_this_nd))
            vtg_at_min = v_tg_axis[min_abs_current_idx]
            iph_at_min = currents_for_this_nd[min_abs_current_idx]
            nd_valley_points.append(nd_level)
            vtg_valley_points.append(vtg_at_min)
            iph_valley_points.append(iph_at_min)
        
        iph_valley_transformed = np.sign(iph_valley_points) * np.log1p(np.abs(iph_valley_points) / linthresh)

        # Create the plot in a new window
        plot_window = tk.Toplevel(self.master)
        plot_window.title(f"Characterization Surface for Pixel {center_pixel_coord}")
        plot_window.geometry("900x800")

        fig = Figure(figsize=(9, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(ND, V_TG, I_PH_TRANSFORMED, cmap='viridis', edgecolor='none', alpha=0.8)
        
        ax.set_xlabel("Optical Power (ND Filter Level)")
        ax.set_ylabel("Gate Voltage (V_tg) [V]")
        ax.set_zlabel(f"Transformed Current\n(sign(I) * ln(1+|I|/{linthresh:.0e}))")
        ax.set_title(f"Device Response Surface - Pixel {center_pixel_coord}", pad=20)
        
        cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label("Transformed Current Value")

        if nd_valley_points:
            ax.plot(nd_valley_points, vtg_valley_points, iph_valley_transformed, 
                    color='r', linewidth=3, marker='o', markersize=6, zorder=10, 
                    label='Valley Trace (Min |I_ph|)')
            ax.legend()
        
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_stability_ratios(self):
        """
        Generates a 2D plot of current ratios and, on a secondary y-axis,
        plots the normalized shape of the ND=2.0 photocurrent curve for context.
        """
        import numpy as np
        import itertools

        # Use the helper function to get the selected pixel's data
        result = self._get_selected_pixel_char()
        if result is None:
            return  # Error message was already shown by the helper
        center_char, center_pixel_coord = result
            
        i_ph_df = center_char.i_ph_vs_nd.copy()
        v_tg_axis = i_ph_df.index.to_numpy()
        
        # Define the ND levels to compare.
        nd_levels_to_compare = [nd for nd in [2.0, 3.0, 4.0] if nd in i_ph_df.columns]

        if len(nd_levels_to_compare) < 2:
            self.update_step_text("Not enough ND levels available for comparison.")
            return

        # Get voltage range from GUI entries
        try:
            vtg_min = float(self.vtg_min_entry.get())
            vtg_max = float(self.vtg_max_entry.get())
            if vtg_min >= vtg_max:
                raise ValueError("Min voltage must be less than max voltage.")
        except ValueError as e:
            self.update_step_text(f"Invalid V_tg range: {e}")
            return

        # Create the plot in a new window
        plot_window = tk.Toplevel(self.master)
        plot_window.title(f"Photocurrent Stability Ratios for Pixel {center_pixel_coord}")
        plot_window.geometry("900x700")

        fig = Figure(figsize=(9, 7), dpi=100)
        ax = fig.add_subplot(111)

        # Manually track Y-axis limits for the ratio curves (left axis)
        y_min_in_range = float('inf')
        y_max_in_range = float('-inf')

        voltage_mask = (v_tg_axis >= vtg_min) & (v_tg_axis <= vtg_max)
        if not np.any(voltage_mask):
            self.update_step_text("The selected V_tg range contains no data points.")
            plot_window.destroy()
            return

        # Plot Ratio Curves (on the primary left Y-axis)
        for nd_a, nd_b in itertools.combinations(nd_levels_to_compare, 2):
            i_numerator = i_ph_df[nd_a].to_numpy()
            i_denominator = i_ph_df[nd_b].to_numpy()

            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.abs(i_numerator) / (np.abs(i_denominator) + 1e-12)
                ratio[np.isnan(ratio)] = 0
            
            visible_ratios = ratio[voltage_mask]
            if visible_ratios.size > 0:
                y_min_in_range = min(y_min_in_range, np.min(visible_ratios))
                y_max_in_range = max(y_max_in_range, np.max(visible_ratios))
            
            ax.plot(v_tg_axis, ratio, label=f'|I(ND={nd_a})| / |I(ND={nd_b})|')

        # --- NEW: Plot Normalized ND=2.0 Current on a Secondary Y-axis ---
        if 2.0 in i_ph_df.columns:
            # Create the secondary axis, sharing the same x-axis
            ax2 = ax.twinx()
            
            i_nd2 = i_ph_df[2.0].to_numpy()
            # Normalize the absolute current to a [0, 1] range for shape comparison
            i_nd2_normalized = np.abs(i_nd2) / (np.max(np.abs(i_nd2)) + 1e-12)
            
            ax2.plot(v_tg_axis, i_nd2_normalized, color='black', linestyle='--', 
                     linewidth=2.5, alpha=0.5, label='Norm. |I(ND=2.0)| Shape')
            
            # Set label for the secondary axis and customize its color
            ax2.set_ylabel("Normalized Current Shape", color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
            # Set the limits for the normalized axis
            ax2.set_ylim(-0.05, 1.05)
        # --- End of Secondary Axis Plot ---

        # Formatting the primary plot (left axis)
        ax.set_xlabel("Gate Voltage (V_tg) [V]")
        ax.set_ylabel("Absolute Photocurrent Ratio", color='C0') # Use a color to distinguish
        ax.tick_params(axis='y', labelcolor='C0')
        ax.set_title(f"Response Stability vs. Gate Voltage for Pixel {center_pixel_coord}")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Set the x-axis limits based on user input
        ax.set_xlim(vtg_min, vtg_max)

        # Set the calculated Y-axis limits for the ratio plot
        if np.isfinite(y_min_in_range) and np.isfinite(y_max_in_range):
            padding = (y_max_in_range - y_min_in_range) * 0.05
            if padding < 1e-9: padding = 0.1 
            ax.set_ylim(y_min_in_range - padding, y_max_in_range + padding)

        # Combine legends from both axes into one
        lines, labels = ax.get_legend_handles_labels()
        if 'ax2' in locals(): # Check if the secondary axis was created
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines.extend(lines2)
            labels.extend(labels2)
        # Add the ideal ratio line last so it's on top in the legend
        ideal_line = ax.axhline(y=1.0, color='r', linestyle=':', linewidth=2, label='Ideal Ratio (y=1.0)')
        lines.append(ideal_line)
        labels.append(ideal_line.get_label())
        
        ax.legend(lines, labels)
        fig.tight_layout() # Adjust layout to make room for the second y-axis

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        
        # Clean up previous plot elements
        if hasattr(self, 'colorbar'): 
            self.colorbar.remove()
            delattr(self, 'colorbar')
        self.output_ax.clear()

        if data is None or data.size == 0:
            self.output_ax.set_xticks([]); self.output_ax.set_yticks([])
            self.output_canvas.draw()
            return
            
        vis_mode = self.vis_mode_combo.get()
        
        # --- Prepare data and settings based on visualization mode ---
        if vis_mode == "Zero Crossings":
            display_data = self._create_zero_crossing_image(data)
            cmap = 'gray'
            vmin, vmax = 0, 255
            colorbar_label = "Crossing Detected"
        elif vis_mode == "Bipolar (RdBu)":
            display_data = data
            cmap = 'RdBu_r'
            vabs_max = np.max(np.abs(data))
            vmin, vmax = -vabs_max if vabs_max > 1e-12 else -1, vabs_max if vabs_max > 1e-12 else 1
            colorbar_label = "Photocurrent (A)"
        else: # Default to Grayscale
            display_data = data
            cmap = 'gray'
            vmin, vmax = data.min(), data.max()
            colorbar_label = "Photocurrent (A)"

        # Ensure vmin and vmax are different to avoid plotting errors
        if abs(vmax - vmin) < 1e-12:
            vmin, vmax = display_data.min() - 0.5, display_data.max() + 0.5
            
        # --- Render the image ---
        im = self.output_ax.imshow(display_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        
        # --- Add a colorbar, but NOT for the binary zero-crossing image ---
        if vis_mode != "Zero Crossings":
            self.colorbar = self.output_fig.colorbar(im, ax=self.output_ax, fraction=0.046, pad=0.04)
            self.colorbar.set_label(colorbar_label)

        self.output_ax.set_xticks([])
        self.output_ax.set_yticks([])
        self.output_fig.tight_layout(pad=0.5)
        self.output_canvas.draw()
        
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