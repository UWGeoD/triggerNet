"""
config.py

Configuration constants for nearest-neighbor clustering of earthquake or acoustic emission events.

Edit these defaults directly, or set/override them programmatically from your main script
(using `config.<PARAM> = value`).

Parameters
----------
B : float or None
    Time exponent (b) for inter-event time normalization. If None, will be estimated.
DF : float
    Fractal dimension of hypocenter distribution (e.g., 1.6 for lab/field events).
Q : float
    Normalization exponent for splitting η into T and R.
ETA0 : float or None
    Threshold for strong links (η₀). If None, will be estimated.
MAG_CUTOFF : float or None
    Magnitude cutoff for events (e.g., 2.0 to include m ≥ 2). If None, include all.

Notes
-----
- These values can be set directly here or at runtime for experiment reproducibility.
- All parameters can be overridden by command-line arguments in the main script.
"""

B: float | None = None      # Will be estimated if not set
DF: float = 1.6
Q: float = 0.5
ETA0: float | None = None   # Will be estimated if not set
MAG_CUTOFF: float | None = None

