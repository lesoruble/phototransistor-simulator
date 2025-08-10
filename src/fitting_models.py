# src/fitting_models.py

import numpy as np

def gaussian_with_baseline(x, slope, intercept, amplitude, center, sigma):
    """ A Gaussian peak on a linear baseline. """
    return (slope * x + intercept) + amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))

def lorentzian_with_baseline(x, slope, intercept, amplitude, center, gamma):
    """ A Lorentzian peak on a linear baseline. gamma is HWHM. """
    return (slope * x + intercept) + amplitude * (gamma**2 / ((x - center)**2 + gamma**2))

def double_gaussian_with_baseline(x, slope, intercept, amp1, cen1, sig1, amp2, cen2, sig2):
    """ A mixture of two Gaussian peaks on a linear baseline. """
    baseline = slope * x + intercept
    gauss1 = amp1 * np.exp(-((x - cen1)**2) / (2 * sig1**2))
    gauss2 = amp2 * np.exp(-((x - cen2)**2) / (2 * sig2**2))
    return baseline + gauss1 + gauss2

def double_lorentzian_with_baseline(x, slope, intercept, amp1, cen1, gam1, amp2, cen2, gam2):
    """ A mixture of two Lorentzian peaks on a linear baseline. """
    baseline = slope * x + intercept
    lorentz1 = amp1 * (gam1**2 / ((x - cen1)**2 + gam1**2))
    lorentz2 = amp2 * (gam2**2 / ((x - cen2)**2 + gam2**2))
    return baseline + lorentz1 + lorentz2

def piecewise_gaussian_with_baseline(x, slope1, slope2, y_at_center, amplitude, center, sigma):
    """ A Gaussian on a V-shaped, piecewise linear baseline. """
    baseline = np.where(x <= center, y_at_center + slope1 * (x - center), y_at_center + slope2 * (x - center))
    gaussian = amplitude * np.exp(-((x - center)**2) / (2 * sigma**2))
    return baseline + gaussian

def calculate_r_squared(y_data, y_fit):
    """Calculates the R-squared value for a fit."""
    ss_res = np.sum((y_data - y_fit)**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    if ss_tot == 0: return 1.0 # Perfect fit or flat line
    return 1 - (ss_res / ss_tot)