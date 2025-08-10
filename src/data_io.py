# src/data_io.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
from src.config import KernelConfig
import logging
import os

log = logging.getLogger(__name__)

@dataclass
class Characterization:
    """Holds all processed data for a single pixel, enabling interpolation."""
    v_tg: np.ndarray          # High-res sorted gate-voltages
    i_ph_vs_nd: pd.DataFrame  # Rows: V_tg, Columns: ND levels, Values: I_ph (HIGH RESOLUTION)
    i_ph_vs_nd_original: pd.DataFrame # Rows: V_tg, Columns: ND levels, Values: I_ph (ORIGINAL SPACING)
    nd_axis: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.nd_axis = self.i_ph_vs_nd.columns.to_numpy(dtype=float)

    def __repr__(self) -> str:
        """Provides a short, human-readable summary for logging."""
        num_points = len(self.v_tg)
        num_nds = len(self.nd_axis)
        return f"<Characterization with {num_points} V_tg points across {num_nds} ND levels>"

# Type alias for the complete data structure: a dictionary mapping (x,y) coords to Characterization objects.
FullCharDataType = Dict[Tuple[int, int], Characterization]

def _load_special_csv(filepath: Path) -> Optional[pd.DataFrame]:
    """Loads a CSV file with a specific commented header format."""
    try:
        with open(filepath, 'r') as f: lines = f.readlines()
        header_line_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('# column names:'): header_line_index = i; break
        if header_line_index == -1: return None
            
        # Extract column names from the line following the marker
        header_line = lines[header_line_index + 1].strip('# \n')
        header = [h.strip() for h in header_line.split(',')]
        
        # Read the rest of the file as data
        df = pd.read_csv(filepath, header=None, names=header, skiprows=header_line_index + 2, skipinitialspace=True, comment='#')
        return df
    except Exception as e:
        log.error(f"Failed to load special CSV '{filepath}': {e}"); return None

def load_all_characterization_from_folder(cfg: KernelConfig) -> Optional[FullCharDataType]:
    """
    Loads all relevant CSV files from a directory, processes them, and returns a
    structured dictionary of per-pixel characterization data.
    """
    # --- 1. Find and Load Raw Data ---
    base_path = Path(__file__).resolve().parent.parent; data_dir = base_path / cfg.data_directory_path
    if not data_dir.is_dir(): log.error(f"Data directory not found: {data_dir}"); return None
    
    csv_files = [f for f in os.listdir(data_dir) if cfg.filename_filter_keyword in f and f.lower().endswith('.csv')]
    if not csv_files: log.error(f"No CSV files with keyword '{cfg.filename_filter_keyword}' found."); return None
    log.info(f"Found {len(csv_files)} matching CSV files.")

    master_df = pd.concat([df for f in csv_files if (df := _load_special_csv(data_dir / f)) is not None], ignore_index=True)
    if master_df.empty: log.error("No data could be loaded from any CSV files."); return None

    # --- 2. Clean, Rename, and Pre-process Data ---
    column_map = {
        cfg.gate1_col: cfg.v_bg_internal, cfg.gate2_col: cfg.v_tg_internal,
        cfg.r_mag_col: "r_lock_in", cfg.phase_col: "phase_lock_in",
        cfg.nd_filter_col: "nd_filter", cfg.x_coord_col: "x_coord", cfg.y_coord_col: "y_coord",
    }
    if not all(col in master_df.columns for col in column_map.keys()): log.error("Missing required columns."); return None
    
    df_proc = master_df[list(column_map.keys())].rename(columns=column_map).copy()
    for col in df_proc.columns: df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')
    df_proc.dropna(inplace=True)
    
    # Calculate the in-phase current from the lock-in amplifier's magnitude and phase
    df_proc[cfg.iph_internal] = df_proc["r_lock_in"] * np.cos(np.deg2rad(df_proc["phase_lock_in"]))
    
    # --- 3. Restructure Data by Pixel and Interpolate ---
    final_data: FullCharDataType = {}
    
    # Create the high-resolution voltage axis for smooth plotting and interpolation
    v_tg_min = df_proc[cfg.v_tg_internal].min()
    v_tg_max = df_proc[cfg.v_tg_internal].max()
    v_tg_len = len(df_proc[cfg.v_tg_internal].unique())
    v_tg_high_res = np.linspace(v_tg_min, v_tg_max, v_tg_len * cfg.interpolation_factor)

    # Group all data points by their (x, y) coordinate
    for pixel_coord, pixel_group in df_proc.groupby(['x_coord', 'y_coord']):
        log.info(f"Processing data for pixel {pixel_coord}")
        
        # Pivot the data to create a grid: V_tg vs ND filter, with I_ph as values.
        # This uses the original, measured V_tg spacing.
        pivot_original = pixel_group.groupby([cfg.v_tg_internal, 'nd_filter'])[cfg.iph_internal].mean().unstack()
        
        # Interpolate to fill any missing V_tg values within each ND curve.
        pivot_interpolated = pivot_original.interpolate(method='index')
        pivot_interpolated.bfill(inplace=True) # Back-fill and forward-fill for robustness
        pivot_interpolated.ffill(inplace=True)
        
        if pivot_interpolated.empty:
            log.warning(f"  -> Could not create valid pivot table for pixel {pixel_coord}. Skipping.")
            continue
            
        # Now create the high-resolution version for smooth plotting
        pivot_hr = pivot_interpolated.reindex(pivot_interpolated.index.union(v_tg_high_res)).interpolate(method='index').loc[v_tg_high_res]
        
        # Store both the high-res and original-spacing data for each pixel
        final_data[pixel_coord] = Characterization(
            v_tg=v_tg_high_res, 
            i_ph_vs_nd=pivot_hr,
            i_ph_vs_nd_original=pivot_interpolated 
        )
        
    if not final_data: log.error("Failed to generate any characterization data."); return None
    return final_data

def normalize_input_image(img: np.ndarray) -> np.ndarray:
    """Normalizes a NumPy array to the range [0.0, 1.0]."""
    if img is None or img.size == 0: return np.array([], dtype=np.float64)
    arr = img.astype(np.float64); mn, mx = arr.min(), arr.max()
    if mx == mn: return np.zeros_like(arr) if mx == 0 else np.ones_like(arr)
    return (arr - mn) / (mx - mn)

@dataclass
class SimpleChar:
    """A temporary, simple characterization class for the optimizer's scaling calculation."""
    i_max_pos: float
    i_max_neg: float