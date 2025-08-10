# In src/cli.py

from __future__ import annotations
import json
import logging
from pathlib import Path
import sys  
import numpy as np
import typer
from PIL import Image
from .config import KernelConfig
from .presets import ALL_PRESET_KERNELS
from .data_io import load_all_characterization_from_folder, normalize_input_image, SimpleChar
from .simulator import Simulator

log = logging.getLogger(__name__)
# Disable Typer's verbose tracebacks for cleaner user-facing error messages.
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _load_kernel(arg: str | Path) -> np.ndarray:
    if isinstance(arg, str) and arg.upper() in ALL_PRESET_KERNELS:
        return ALL_PRESET_KERNELS[arg.upper()].astype(np.float32)

    p = Path(arg)
    if not p.exists():
        raise typer.BadParameter(f"Kernel '{arg}' not found.")

    if p.suffix.lower() == ".json":
        arr = np.array(json.loads(p.read_text()), dtype=np.float32)
    else:  # assume .npy
        arr = np.load(p).astype(np.float32)

    if arr.shape != (3, 3):
        raise typer.BadParameter("Kernel must be 3×3.")
    return arr


# In cli.py, replace the entire run_kernel_cmd function.

@app.command("run-kernel")
def run_kernel_cmd(
    image_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to the input image file."),
    kernel: str = typer.Option("IDENTITY", help="Preset name (e.g., 'SOBEL_X') or path to a .json/.npy kernel file."),
    output: str = typer.Option(None, "--output", "-o", help="Output PNG path. Defaults to '<image_name>_conv.png'."),
    verbose: bool = typer.Option(False, "--verbose/--quiet", "-v", help="Enable detailed INFO-level logging."),
    nd_level: float = typer.Option(2.0, help="The ND filter level to use for the 'Fixed Power' simulation."),
) -> None:
    """
    Performs batch-mode convolution equivalent to the GUI's 'Fixed Power' mode.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    try:
        # --- 1. Load Data and Initialize Simulator ---
        cfg = KernelConfig()
        char_data = load_all_characterization_from_folder(cfg)
        if not char_data:
            print("Error: Could not load characterization data. Aborting.", file=sys.stderr)
            raise typer.Exit(code=1)
        
        simulator = Simulator(char_data, cfg)
        unitless_kernel = _load_kernel(kernel)

        # --- 2. Calculate Kernel Weights ---
        log.info(f"Calculating kernel for fixed power mode at ND={nd_level}...")
        fixed_power_result = simulator.run_fixed_power_calculation(
            unitless_kernel, nd_level, nd5_voltage_cache=None, preset_name=kernel.upper()
        )
        kernel_weights_amps = fixed_power_result.currents
        log.info(f"Calculated Kernel Weights (Amps):\n{kernel_weights_amps}")

        # --- 3. Run Convolution ---
        img_np = np.asarray(Image.open(image_path).convert("L"))
        norm_img = normalize_input_image(img_np)
        
        raw_output = simulator.run_convolution(norm_img, kernel_weights_amps)

        # --- 4. Normalize and Save Output ---
        mn, mx = raw_output.min(), raw_output.max()
        if mx - mn > 1e-12:
            norm_out = (raw_output - mn) / (mx - mn)
        else:
            norm_out = np.zeros_like(raw_output)
        
        out_img = Image.fromarray((norm_out * 255).astype(np.uint8))

        if output is None:
            output_path = image_path.with_name(f"{image_path.stem}_conv_{kernel.lower()}_nd{nd_level}.png")
        else:
            output_path = Path(output)

        out_img.save(output_path)
        log.info(f"Successfully saved output to {output_path}")

    except Exception as e:
        print(f"An error occurred during execution: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        raise typer.Exit(code=1)

def cli_entry() -> None:
    app()

if __name__ == "__main__":
    cli_entry()