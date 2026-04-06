"""
preprocessing.py
----------------
Missing data handling, normalization, and baseline correction.
Applies signal-type-aware preprocessing to cleaned DataFrames.
"""

import numpy as np
import pandas as pd


def baseline_correction(signal: np.ndarray) -> np.ndarray:
    """Remove baseline offset by subtracting the signal mean."""
    return signal - np.mean(signal)


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize signal to the range [0, 1]."""
    sig_min = np.min(signal)
    sig_max = np.max(signal)

    if sig_max == sig_min:
        return np.zeros_like(signal)

    return (signal - sig_min) / (sig_max - sig_min)


def apply_preprocessing(df: pd.DataFrame, signal_type: str) -> pd.DataFrame:
    """
    Apply preprocessing based on signal type.

    ECG:         baseline correction only (preserve peak amplitudes)
    Temperature:  baseline correction + normalization
    Respiration: baseline correction + normalization
    Motion:      baseline correction + normalization
    """
    df = df.copy()
    signal = df["signal"].to_numpy()

    signal = baseline_correction(signal)

    if signal_type != "ECG":
        signal = normalize_signal(signal)

    df["signal"] = signal
    return df
