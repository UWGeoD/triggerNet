#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:14:49 2025

@author: ellie

config.py

Configuration constants for nearest-neighbor earthquake clustering analysis.
"""

# Time exponent (b) for inter-event time normalization
B = None #1.0

# Fractal dimension of hypocenter distribution (df)
DF = 1.6

# Normalization exponent (q) for splitting η into T and R
Q = 0.5

# Separation threshold for nearest-neighbor distance (η₀) in log10 units
ETA0 = None #1e-5

# Magnitude cutoff for events to include in analysis (e.g., m ≥ 2)
MAG_CUTOFF = None
