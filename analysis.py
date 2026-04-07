"""
analysis.py
-----------
Statistics, FFT computation, and signal-type-specific feature extraction.
"""

import numpy as np
from scipy.signal import find_peaks




def compute_statistics(signal: np.ndarray) -> dict:
    """
    Compute basic signal statistics:
    mean, standard deviation, RMS, peak-to-peak range.
    """
    return {
        "Mean": float(np.mean(signal)),
        "Std Dev": float(np.std(signal)),
        "RMS": float(np.sqrt(np.mean(signal ** 2))),
        "Peak-to-Peak": float(np.ptp(signal)),
    }




def compute_fft(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute single-sided FFT magnitude spectrum.
    Returns (frequencies, magnitudes).
    """
    n = len(signal)
    fft_values = np.fft.rfft(signal)
    magnitudes = np.abs(fft_values) / n
    frequencies = np.fft.rfftfreq(n, d=1.0 / fs)
    return frequencies, magnitudes


def find_dominant_frequencies(frequencies: np.ndarray,
                              magnitudes: np.ndarray,
                              num_peaks: int = 3) -> list[dict]:
    """
    Identify the top *num_peaks* dominant frequency components.
    Returns list of {frequency, magnitude} dicts sorted by magnitude desc.
    """
    if len(magnitudes) < 3:
        return []

    
    mags = magnitudes[1:].copy()
    freqs = frequencies[1:].copy()

    peaks, properties = find_peaks(mags, height=0)

    if len(peaks) == 0:
       
        idx = np.argmax(mags)
        return [{"frequency": float(freqs[idx]),
                 "magnitude": float(mags[idx])}]

    heights = properties["peak_heights"]
    sorted_indices = np.argsort(heights)[::-1][:num_peaks]

    results = []
    for i in sorted_indices:
        peak_idx = peaks[i]
        results.append({
            "frequency": float(freqs[peak_idx]),
            "magnitude": float(mags[peak_idx]),
        })
    return results




def extract_features(signal: np.ndarray, time: np.ndarray,
                     signal_type: str, fs: float) -> dict:
    """
    Extract features tailored to the selected signal type.
    """
    features = {}

    if signal_type == "ECG":
        peaks, _ = find_peaks(
            signal,
            distance=int(0.5 * fs),
            prominence=np.std(signal) * 0.5,
        )
        duration = time[-1] - time[0] if len(time) > 1 else 0
        bpm = (len(peaks) / duration) * 60 if duration > 0 else 0.0
        features["Estimated Heart Rate (BPM)"] = round(bpm, 2)
        features["Detected R-Peaks"] = int(len(peaks))

    elif signal_type == "Temperature":
        if len(time) > 1:
            coeffs = np.polyfit(time, signal, 1)
            slope = coeffs[0]
        else:
            slope = 0.0
        features["Trend Slope"] = round(float(slope), 6)
        features["Average Value"] = round(float(np.mean(signal)), 4)
        features["Variance"] = round(float(np.var(signal)), 6)

    elif signal_type == "Respiration":
        peaks, _ = find_peaks(
            signal,
            distance=max(1, int(1.5 * fs)),
            prominence=np.std(signal) * 0.3,
        )
        duration = time[-1] - time[0] if len(time) > 1 else 0
        rate = (len(peaks) / duration) * 60 if duration > 0 else 0.0
        features["Breathing Rate (breaths/min)"] = round(rate, 2)
        features["Detected Breaths"] = int(len(peaks))
        features["RMS Amplitude"] = round(float(np.sqrt(np.mean(signal ** 2))), 4)

    elif signal_type == "Motion":
        features["Peak Acceleration"] = round(float(np.max(np.abs(signal))), 4)
        features["Activity RMS"] = round(float(np.sqrt(np.mean(signal ** 2))), 4)

    return features
