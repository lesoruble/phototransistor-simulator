# src/image_comparator.py

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os

def load_csv_as_image(filepath: str) -> np.ndarray:
    """Loads a CSV file into a NumPy array, ensuring it's treated as an 8-bit image."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at '{filepath}'")
    return np.loadtxt(filepath, delimiter=',', dtype=np.uint8)

def calculate_metrics(img1: np.ndarray, img2: np.ndarray) -> dict:
    """Calculates a dictionary of comparison metrics between two images."""
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "PSNR (dB)": psnr(img1, img2, data_range=255),
        "SSIM": ssim(img1, img2, data_range=255, channel_axis=None)
    }
