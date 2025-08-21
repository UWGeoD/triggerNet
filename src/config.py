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
ETA0 : float or None
    Threshold for strong links (η₀). If None, will be estimated.
MAG_CUTOFF : float or None
    Magnitude cutoff for events (e.g., 2.0 to include m ≥ 2). If None, include all.
VELOCITY : float or None
    Maximum allowed propagation velocity for event connection (distance/time). If None, no velocity constraint is applied.

Notes
-----
- These values can be set directly here or at runtime for experiment reproducibility.
- All parameters can be overridden by command-line arguments in the main script.
"""

B: float | None = None      # Will be estimated if not set
DF: float = 1.6
ETA0: float | None = None   # Will be estimated if not set
MAG_CUTOFF: float | None = None
VELOCITY: float | None = None