#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:15:54 2025

@author: ellie

data_io.py

Functions to load, parse, and filter earthquake catalogs.
"""

import pandas as pd
from datetime import datetime
import config
import scipy
import os

def load_catalog(
    file_path: str,
    time_col: str = 'time',
    x_col: str = 'x',
    y_col: str = 'y',
    mag_col: str = 'mag',
    z_col: str = None,
    time_format: str = None
) -> pd.DataFrame:
    """
    Load an earthquake catalog from CSV and filter by magnitude.

    Args:
        file_path: Path to the CSV catalog.
        time_col: Name of the time column (to be parsed).
        lat_col: Name of the latitude (or x) column.
        lon_col: Name of the longitude (or y) column.
        mag_col: Name of the magnitude column.
        depth_col: Optional name of the depth (or z) column.
        time_format: Optional datetime format for parsing.
        is_lab: If True, treat (lat_col, lon_col, depth_col) as 3D Euclidean coords.

    Returns:
        Filtered, time-sorted DataFrame.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext == '.mat':
        raw = scipy.io.loadmat(file_path)
        # Filter out meta keys like __header__, __globals__, etc.
        data = {}
        for k, v in raw.items():
            if k.startswith("__"):
                continue
            v_squeezed = v.squeeze()
            if v_squeezed.ndim == 1:
                data[k] = v_squeezed
            else:
                print(f"Skipping key '{k}' with shape {v.shape} (not 1D)")
        df = pd.DataFrame(data)

    else:
        raise ValueError("Unsupported file format. Use .csv or .mat")

    # Parse times
    if time_format:
        df[time_col] = pd.to_datetime(df[time_col], format=time_format)

    # Standardize columns
    df = df.rename(columns={
        time_col: 'time',
        x_col: 'x',
        y_col: 'y',
        mag_col: 'mag'
    })
    if z_col:
        df = df.rename(columns={z_col: 'z'})

    # Filter by magnitude
    if config.MAG_CUTOFF:
        df = df[df['mag'] >= config.MAG_CUTOFF].copy()
    df = df.sort_values('time').reset_index(drop=True)
    return df
