"""
data_io.py

Utility functions for loading, parsing, and filtering earthquake or acoustic emission event catalogs.

Supports:
    - CSV files (standard pandas format)
    - MATLAB .mat files (1D arrays with column names)

Standardizes column names to ['time', 'x', 'y', 'mag', (optional) 'z'] and applies magnitude cutoff.
"""

import pandas as pd
import numpy as np
import os
import scipy.io
import config

def load_catalog(
    file_path: str,
    time_col: str = 'time',
    x_col: str = 'x',
    y_col: str = 'y',
    mag_col: str = 'mag',
    z_col: str = None,
    time_format: str = None,
) -> pd.DataFrame:
    """
    Load an earthquake or AE event catalog from CSV or MAT file.
    Standardize column names, parse times, and filter by magnitude.

    Args:
        file_path (str): Path to input .csv or .mat catalog.
        time_col (str): Name of time column in file.
        x_col (str): Name of x (or longitude/latitude) column in file.
        y_col (str): Name of y (or latitude/longitude) column in file.
        mag_col (str): Name of magnitude column in file.
        z_col (str or None): Optional name of depth/z column.
        time_format (str or None): Optional datetime format for time parsing (for CSV).

    Returns:
        pd.DataFrame: Filtered and time-sorted catalog with standardized columns.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # --- Load Data ---
    if ext == '.csv':
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file '{file_path}': {e}")
    elif ext == '.mat':
        try:
            raw = scipy.io.loadmat(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load MAT file '{file_path}': {e}")

        # Convert .mat structure to DataFrame (flattened 1D arrays)
        data = {}
        for k, v in raw.items():
            if k.startswith("__"):
                continue  # meta keys
            v = np.atleast_1d(v).squeeze()
            if v.ndim == 1:
                data[k] = v
            else:
                print(f"[WARN] Skipping key '{k}' with shape {v.shape} (not 1D)")
        if not data:
            raise ValueError(f"No valid data arrays found in MAT file '{file_path}'.")
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Only .csv and .mat are supported.")

    # --- Parse and Standardize Columns ---
    for req_col in [time_col, x_col, y_col, mag_col]:
        if req_col not in df.columns:
            raise ValueError(f"Required column '{req_col}' not found in input file: {file_path}")

    rename_dict = {
        time_col: 'time',
        x_col: 'x',
        y_col: 'y',
        mag_col: 'mag'
    }
    if z_col and z_col in df.columns:
        rename_dict[z_col] = 'z'

    df = df.rename(columns=rename_dict)

    # --- Parse Time Column ---
    if time_format:
        try:
            df['time'] = pd.to_datetime(df['time'], format=time_format).astype('int64').to_numpy() / 1e9
        except Exception as e:
            raise RuntimeError(f"Failed to parse time column with format '{time_format}': {e}")

    # --- Filter by Magnitude Cutoff ---
    if getattr(config, "MAG_CUTOFF", None) is not None:
        before = len(df)
        df = df[df['mag'] >= config.MAG_CUTOFF].copy()
        after = len(df)
        if after < before:
            print(f"[INFO] Filtered {before-after} events below magnitude cutoff {config.MAG_CUTOFF}.")

    # --- Sort by Time ---
    df = df.sort_values('time').reset_index(drop=True)

    return df
