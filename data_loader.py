"""
data_loader.py
--------------
CSV reading, validation, and cleaning of sensor data files.
Handles missing/corrupted values via interpolation and fill methods.
"""

import pandas as pd
import numpy as np


VALID_SIGNAL_TYPES = ["ECG", "Temperature", "Respiration", "Motion"]


def load_csv_file(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    Raises informative errors for missing, empty, or malformed files.
    """
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("The CSV file is empty (no data rows).")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty (no content at all).")
    except pd.errors.ParserError:
        raise ValueError("The CSV file format is invalid or corrupted.")


def validate_and_clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Validate CSV structure and clean bad or missing data.
    Returns the cleaned DataFrame and a report dict with cleaning details.

    Assumes:
      - Column 1 = time
      - Column 2 = signal amplitude
    """
    report = {}

    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least 2 columns (time and signal).")

    # Keep first two columns only
    df = df.iloc[:, :2].copy()
    df.columns = ["time", "signal"]

    # Coerce to numeric — non-numeric entries become NaN
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce")

    report["missing_time"] = int(df["time"].isna().sum())
    report["missing_signal"] = int(df["signal"].isna().sum())
    report["total_rows_before"] = len(df)

    # Drop rows where time is missing (cannot interpolate time reliably)
    df = df.dropna(subset=["time"])

    # Interpolate missing signal values linearly, then back/forward fill edges
    df["signal"] = df["signal"].interpolate(method="linear")
    df["signal"] = df["signal"].bfill().ffill()

    if df["signal"].isna().any():
        raise ValueError("Signal column still contains invalid values after cleaning.")

    if df.empty:
        raise ValueError("No valid data remains after cleaning.")

    report["total_rows_after"] = len(df)
    report["rows_removed"] = report["total_rows_before"] - report["total_rows_after"]

    return df, report


def estimate_sampling_frequency(df: pd.DataFrame) -> float:
    """
    Estimate sampling frequency from the time column (assumes seconds).
    """
    dt = np.diff(df["time"].to_numpy())
    dt_mean = np.mean(dt)

    if dt_mean <= 0:
        raise ValueError("Invalid time column — time must be increasing.")

    return 1.0 / dt_mean
