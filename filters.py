"""
filters.py
----------
Digital filtering: LPF, HPF, BPF using both FIR and IIR (Butterworth).
Includes default cutoff-frequency tables keyed by signal type.
"""

import numpy as np
from scipy.signal import butter, filtfilt, firwin




DEFAULT_LOWPASS_CUTOFF = {
    "ECG": 40.0,
    "Temperature": 0.1,
    "Respiration": 2.0,
    "Motion": 10.0,
}

DEFAULT_HIGHPASS_CUTOFF = {
    "ECG": 0.5,
    "Temperature": 0.005,
    "Respiration": 0.1,
    "Motion": 0.5,
}

DEFAULT_BANDPASS_CUTOFFS = {
    "ECG": (0.5, 40.0),
    "Temperature": (0.005, 0.1),
    "Respiration": (0.1, 2.0),
    "Motion": (0.5, 10.0),
}


def get_default_lowpass_cutoff(signal_type: str) -> float:
    return DEFAULT_LOWPASS_CUTOFF.get(signal_type, 10.0)


def get_default_highpass_cutoff(signal_type: str) -> float:
    return DEFAULT_HIGHPASS_CUTOFF.get(signal_type, 0.5)


def get_default_bandpass_cutoffs(signal_type: str) -> tuple[float, float]:
    return DEFAULT_BANDPASS_CUTOFFS.get(signal_type, (0.5, 10.0))




def _validate_cutoff(cutoff: float, nyquist: float, label: str = "Cutoff"):
    if cutoff <= 0:
        raise ValueError(f"{label} frequency must be positive.")
    if cutoff >= nyquist:
        raise ValueError(
            f"{label} frequency ({cutoff:.2f} Hz) must be less than "
            f"Nyquist frequency ({nyquist:.2f} Hz)."
        )



def lowpass_iir(signal: np.ndarray, fs: float, cutoff: float,
                order: int = 4) -> np.ndarray:
    """IIR Butterworth low-pass filter."""
    nyquist = 0.5 * fs
    _validate_cutoff(cutoff, nyquist, "Low-pass cutoff")
    b, a = butter(order, cutoff / nyquist, btype="low")
    return filtfilt(b, a, signal)


def highpass_iir(signal: np.ndarray, fs: float, cutoff: float,
                 order: int = 4) -> np.ndarray:
    """IIR Butterworth high-pass filter."""
    nyquist = 0.5 * fs
    _validate_cutoff(cutoff, nyquist, "High-pass cutoff")
    b, a = butter(order, cutoff / nyquist, btype="high")
    return filtfilt(b, a, signal)


def bandpass_iir(signal: np.ndarray, fs: float, low: float, high: float,
                 order: int = 4) -> np.ndarray:
    """IIR Butterworth band-pass filter."""
    nyquist = 0.5 * fs
    if low >= high:
        raise ValueError("Low cutoff must be less than high cutoff.")
    _validate_cutoff(low, nyquist, "Low cutoff")
    _validate_cutoff(high, nyquist, "High cutoff")
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, signal)




def _safe_numtaps(numtaps: int, signal_length: int) -> int:
    """
    Ensure numtaps is compatible with filtfilt's padlen requirement.
    filtfilt needs: signal_length > 3 * numtaps - 1
    We also keep numtaps odd (required by firwin for HPF / BPF).
    """
    max_taps = (signal_length - 1) // 3
    if max_taps < 3:
        raise ValueError(
            f"Signal is too short ({signal_length} samples) for FIR filtering. "
            f"Need at least 10 samples."
        )
    safe = min(numtaps, max_taps)
    
    if safe % 2 == 0:
        safe -= 1
    return max(safe, 3)




def lowpass_fir(signal: np.ndarray, fs: float, cutoff: float,
                numtaps: int = 101) -> np.ndarray:
    """FIR low-pass filter using a windowed sinc (Hamming)."""
    nyquist = 0.5 * fs
    _validate_cutoff(cutoff, nyquist, "Low-pass cutoff")
    numtaps = _safe_numtaps(numtaps, len(signal))
    taps = firwin(numtaps, cutoff / nyquist, pass_zero="lowpass")
    return filtfilt(taps, [1.0], signal)


def highpass_fir(signal: np.ndarray, fs: float, cutoff: float,
                 numtaps: int = 101) -> np.ndarray:
    """FIR high-pass filter using a windowed sinc (Hamming)."""
    nyquist = 0.5 * fs
    _validate_cutoff(cutoff, nyquist, "High-pass cutoff")
    numtaps = _safe_numtaps(numtaps, len(signal))
    taps = firwin(numtaps, cutoff / nyquist, pass_zero=False)
    return filtfilt(taps, [1.0], signal)


def bandpass_fir(signal: np.ndarray, fs: float, low: float, high: float,
                 numtaps: int = 101) -> np.ndarray:
    """FIR band-pass filter using a windowed sinc (Hamming)."""
    nyquist = 0.5 * fs
    if low >= high:
        raise ValueError("Low cutoff must be less than high cutoff.")
    _validate_cutoff(low, nyquist, "Low cutoff")
    _validate_cutoff(high, nyquist, "High cutoff")
    numtaps = _safe_numtaps(numtaps, len(signal))
    taps = firwin(numtaps, [low / nyquist, high / nyquist], pass_zero=False)
    return filtfilt(taps, [1.0], signal)




FILTER_FUNCTIONS = {
    "LPF_IIR": lowpass_iir,
    "HPF_IIR": highpass_iir,
    "BPF_IIR": bandpass_iir,
    "LPF_FIR": lowpass_fir,
    "HPF_FIR": highpass_fir,
    "BPF_FIR": bandpass_fir,
}


def apply_filter(signal: np.ndarray, fs: float, filter_key: str,
                 signal_type: str,
                 custom_cutoff: float | None = None,
                 custom_low: float | None = None,
                 custom_high: float | None = None,
                 order: int = 4,
                 numtaps: int = 101) -> np.ndarray:
    """
    Central dispatcher: apply the chosen filter with either user-supplied
    or default parameters.
    """
    is_bandpass = filter_key.startswith("BPF")
    is_fir = filter_key.endswith("FIR")

    if is_bandpass:
        default_low, default_high = get_default_bandpass_cutoffs(signal_type)
        low = custom_low if custom_low is not None else default_low
        high = custom_high if custom_high is not None else default_high
        if is_fir:
            return bandpass_fir(signal, fs, low, high, numtaps)
        else:
            return bandpass_iir(signal, fs, low, high, order)
    else:
        if filter_key.startswith("LPF"):
            default_cutoff = get_default_lowpass_cutoff(signal_type)
        else:
            default_cutoff = get_default_highpass_cutoff(signal_type)

        cutoff = custom_cutoff if custom_cutoff is not None else default_cutoff

        func = FILTER_FUNCTIONS[filter_key]
        if is_fir:
            return func(signal, fs, cutoff, numtaps)
        else:
            return func(signal, fs, cutoff, order)