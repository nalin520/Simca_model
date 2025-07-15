
import numpy as np
from scipy.signal import savgol_filter

def savgol_smooth(y, window=15, polyorder=3):
    """Savitzky-Golay smoothing with safety checks."""
    window = min(len(y)//2*2+1, window)          
    if window < polyorder + 2:
        return y
    return savgol_filter(y, window, polyorder)

def median_MAD_scale(y, eps=1e-9):
    """Robust per-spectrum scaling: (y - median) / MAD."""
    med = np.median(y)
    mad = np.median(np.abs(y - med)) + eps
    return (y - med) / mad