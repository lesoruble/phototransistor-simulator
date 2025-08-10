# src/config.py

from dataclasses import dataclass, field
import numpy as np
from typing import Dict

@dataclass(slots=True)
class KernelConfig:
    """
    Stores all tunable, environment-specific constants used by the application.
    Defaults are set to match the original behavior.
    """
    # --- Data Source Configuration ---
    data_directory_path: str = "250624_SlothGUI/"
    filename_filter_keyword: str = "KernelPhotoTopGating"
    interpolation_factor: int = 10 # How many times more points to create

    # --- Column Name Mapping from CSV ---
    gate1_col: str = "Gate Voltage"
    gate2_col: str = "Gate Voltage 2"
    r_mag_col: str = "R Lock_in"
    phase_col: str = "Phase Lock_in"
    nd_filter_col: str = "ND Filter"
    x_coord_col: str = "X Coord"
    y_coord_col: str = "Y Coord"
    
    # --- Internal Column Names (used after loading) ---
    v_bg_internal: str = "v_bg"
    v_tg_internal: str = "v_tg"
    iph_internal: str = "in_phase_current"

    # --- GUI & Processing Parameters ---
    decimals_v_tg_display: int = 2
    fixed_vbg_value: float = -1.2 
    fallback_kernel_scale_factor: float = 5e-8 # Used if characterization data is missing

    # --- Dynamic Regime Search Parameters ---
    REFERENCE_PIXEL_X: int = 2
    REFERENCE_PIXEL_Y: int = 2
    # The ND filter levels to test when finding the optimal fixed voltages.
    REFERENCE_ND_LEVELS_TO_TEST: list[float] = field(default_factory=lambda: [2.0, 3.0, 4.0])
    # How many V_tg steps to the left and right of the anchor to test.
