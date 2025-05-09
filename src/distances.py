#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 12:20:15 2025

@author: ellie

distances.py

Compute inter-event distances (raw η and normalized components T,R) for nearest-neighbor clustering.
"""

import numpy as np
from datetime import datetime
from math import radians, sin, cos, sqrt
from datetime import datetime
import config

def haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points on the Earth specified in decimal degrees.
    Returns distance in kilometers.
    """
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * np.arcsin(sqrt(a))
    # Earth radius ~6371 km
    return 6371.0 * c

def euclidean3d(i, j):
    """Simple 3D distance on x,y,z."""
    dx = i['x'] - j['x']
    dy = i['y'] - j['y']
    dz = i.get('z', 0.0) - j.get('z', 0.0)
    return sqrt(dx*dx + dy*dy + dz*dz)


def compute_eta(t_i, t_j, r, m_i):
    """
    Vectorized η = r^DF / (10^(B * m_i) * (t_j - t_i))
    Returns +inf where (t_j - t_i) <= 0.
    """
    t_diff = t_j - t_i
    # wherever t_diff <= 0, we want η = +inf, else compute formula
    eta = np.where(
        t_diff > 0,
        (r ** config.DF) / (10 ** (config.B * m_i) * t_diff),
        np.inf
    )
    return eta


def compute_TR(t_i, t_j, r, M_i):
    """
    Compute normalized time (T) and spatial (R) components:
      R = (r^Df) * sqrt(M_i)
      T = (Δt)  * sqrt(M_i)
    where Δt = t_j − t_i.
    Args:
        t_i (float): time of precursor event
        t_j (float): time of current event
        r   (float): distance between events
        M_i (float): precomputed moment weight = 10^(−B * mag_i)
    Returns:
        (T, R)
    """
    delta_t = t_j - t_i
    sqrt_M  = np.sqrt(M_i)
    T = delta_t * sqrt_M
    R = (r ** config.DF) * sqrt_M
    return T, R