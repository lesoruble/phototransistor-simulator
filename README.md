# Adaptive Phototransistor Kernel Simulator

This project provides a Python-based GUI and command-line tool to simulate the convolution of an image using a 3x3 kernel implemented on a physical array of phototransistors. The key feature is its ability to operate in two modes:

1.  **Fixed Power Mode:** Simulates the device array when illuminated by a uniform, fixed optical power (specified by an ND filter value).
2.  **Dynamic Mode:** A more advanced simulation where the kernel's behavior adapts based on the local intensity of the input image. This is achieved by using a fixed set of gate voltages, allowing the phototransistor responses to vary naturally with incident light.

The application loads real-world characterization data from CSV files to model the behavior of each phototransistor pixel.

## Features

-   **Interactive GUI:** A Tkinter-based interface for easy control and visualization.
-   **Dual-Mode Operation:** Supports both "Fixed Power" and "Dynamic Mode" simulations.
-   **Automated Mode Comparison:** A one-click tool to convolve an image in both modes and display a visual comparison with quality metrics (MSE, RMSE, PSNR, SSIM).
-   **Live Kernel Visualization:** See the applied gate voltages and resulting photocurrents.
-   **Step-by-Step Convolution:** Walk through the convolution process one pixel at a time to understand the underlying calculations.
-   **Preset Kernels:** Includes standard image processing kernels like Identity, Sobel, Laplacian, etc.
-   **Custom Kernel Weights:** Manually define your own unitless kernel weights.
-   **Command-Line Interface (CLI):** A `typer`-based CLI for batch processing and scripting.
-   **Data Export:** Save convolution results as a PNG image or a CSV file (in either raw current values or normalized grayscale).
-   **Dynamic Response Plots:** Visualize how each kernel pixel's current responds to light in the "Dynamic Mode".

## Project Structure

```
adaptive-phototransistor-simulator/
├── scripts/
│   └── precompute_nd5.py     # Caching script for performance
├── 250624_SlothGUI/      # Directory for characterization data (ignored by Git)
├── run_gui.py            # Main entry point to launch the GUI
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── src/
    ├── __init__.py       # Makes `src` a package
    ├── cli.py            # Command-line interface logic
    ├── config.py         # Configuration constants
    ├── data_io.py        # Data loading and processing
    ├── fitting_models.py # Mathematical models for fitting
    ├── gui.py            # The main application GUI class
    ├── image_comparator.py # Logic for image quality metrics
    ├── kernel.py         # Core device physics calculations
    ├── presets.py        # Preset kernel definitions
    └── simulator.py      # Core calculation engine
```

## Setup and Installation

1.  **Clone the repository and navigate into the project directory:**
    ```bash
    git clone https://github.com/lesoruble/phototransistor-simulator.git
    cd phototransistor-simulator
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data:**
    Ensure the characterization data folder (e.g., `250624_SlothGUI/`) is present in the project's root directory. The path can be configured in `src/config.py`.

## Pre-computation for Performance (Required First Step)

Before running the main application for the first time, you must generate the voltage cache file. This script analyzes the device data and optimizes calculations for the GUI. Run the following command from your project's root directory:
```bash
python scripts/precompute_nd5.py
```

## How to Run

## Analysis Scripts

The `scripts/` directory contains additional Python scripts for performing detailed data analysis and generating plots. You can run them directly after setting up your environment.

-   **`scripts/plotter_csv.py`**: Analyzes double-gated measurements to find the optimal back-gate voltage (`V_BG`) for maximizing photocurrent swing.
-   **`scripts/plotter_pixels_bypower.py`**: Creates detailed plots of photocurrent vs. top-gate voltage for each pixel, grouped by optical power (ND filter). Includes advanced curve fitting for noisy data.

**Example:**
```bash
python scripts/plotter_csv.py
```

### GUI Application

To launch the graphical user interface, run the `run_gui.py` script:
```bash
python run_gui.py
```

### Command-Line Interface (CLI)

The CLI allows for non-interactive, scriptable convolution.

**Usage:**
```bash
python -m src.cli run-kernel [OPTIONS] IMAGE_PATH
```

**Example:**
Convolve `my_image.png` with the `SOBEL_X` kernel at `ND=3.0`:
```bash
python -m src.cli run-kernel my_image.png --kernel SOBEL_X --nd-level 3.0
```

**Get help:**
For a full list of options, use the `--help` flag:
```bash
python -m src.cli run-kernel --help
```