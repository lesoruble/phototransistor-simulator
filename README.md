# Adaptive Phototransistor Kernel Simulator

This project provides a Python-based GUI and command-line tool to simulate the convolution of an image using a 3x3 kernel implemented on a physical array of phototransistors. The key feature is its ability to operate in two modes:

1.  **Fixed Power Mode:** Simulates the device array when illuminated by a uniform, fixed optical power (specified by an ND filter value).
2.  **Dynamic Mode:** A more advanced simulation where the kernel's behavior adapts based on the local intensity of the input image. This is achieved by using a fixed set of gate voltages, allowing the phototransistor responses to vary naturally with incident light.

The application loads real-world characterization data from CSV files to model the behavior of each phototransistor pixel.

## Features

-   **Interactive GUI:** A Tkinter-based interface for easy control and visualization.
-   **Dual-Mode Operation:** Supports both "Fixed Power" and "Dynamic Mode" simulations.
-   **Live Kernel Visualization:** See the applied gate voltages and resulting photocurrents.
-   **Step-by-Step Convolution:** Walk through the convolution process one pixel at a time to understand the underlying calculations.
-   **Preset Kernels:** Includes standard image processing kernels like Identity, Sobel, Laplacian, etc.
-   **Custom Kernel Weights:** Manually define your own unitless kernel weights.
-   **Command-Line Interface (CLI):** A `typer`-based CLI for batch processing and scripting.
-   **Data Export:** Save convolution results as a PNG image or a CSV file (in either raw current values or normalized grayscale).
-   **Dynamic Response Plots:** Visualize how each kernel pixel's current responds to light in the "Dynamic Mode".

## Project Structure
.
├── 250624_SlothGUI/ # Directory for characterization data (not included in repo)
├── run_gui.py # Main entry point to launch the GUI
├── requirements.txt # Python dependencies
├── README.md # This file
└── src/
├── cli.py # Command-line interface logic
├── config.py # Configuration constants and parameters
├── data_io.py # Data loading and processing from CSVs
├── fitting_models.py # Mathematical models for data fitting
├── gui.py # The main application GUI class and logic
├── kernel.py # Core convolution and device physics calculations
└── presets.py # Preset kernel definitions

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <repository-folder>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data:**
    Ensure the characterization data folder (e.g., `250624_SlothGUI/`) is present in the project's root directory. The path can be configured in `src/config.py`.

## Pre-computation for Performance (Required First Step)

The simulator uses a cache file (`nd5_voltage_cache.json`) to significantly speed up calculations in the GUI when operating in the noisy `ND=5` fixed-power mode. This cache must be generated once before running the application for the first time.

The `precompute_nd5.py` script performs this task. It analyzes the characterization data, fits models to the noisy `ND=5` curves, and calculates the optimal gate voltages for each preset kernel.

To generate the cache, run the following command from the project root:
    ```bash
    python precompute_nd5.py

## How to Run

### GUI Application

To launch the graphical user interface, run the `run_gui.py` script from the project root directory:
    ```bash
    python run_gui.py

### Command-Line Interface (CLI)
The CLI allows for non-interactive, scriptable convolution.
# Usage:
    python -m src.cli run-kernel [OPTIONS] IMAGE_PATH
# Example:
Convolve my_image.png with the SOBEL_X preset kernel and save the output to result.png.
    python -m src.cli run-kernel my_image.png --kernel SOBEL_X --output result.png
# Get help:
For a full list of options, use the --help flag.
    python -m src.cli run-kernel --help
