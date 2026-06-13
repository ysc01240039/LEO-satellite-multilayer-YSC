#!/usr/bin/env python3
"""
Export orbit data from parquet to binary files for C++ simulation.

C++ expected format (matching main.cpp load_orbit):
    - 4 bytes: int (number of time points)
    - Then n_points × 3 doubles: x, y, z (positions only, in km)

This script MUST be run once before the C++ simulation.
"""

import os
import json
import struct
import pandas as pd
import numpy as np

# Load satellite metadata
with open("metadata.json", 'r') as f:
    satellites = json.load(f)

OUTPUT_DIR = "orbit_bin"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Exporting {len(satellites)} satellites to {OUTPUT_DIR}/...")
print(f"Format: int(n_points) + n_points * (3 doubles: x, y, z)")

for sat in satellites:
    sat_id = sat['sat_id']
    parquet_file = sat['parquet_file']
    
    df = pd.read_parquet(parquet_file)
    df = df.sort_values('t')
    
    x = df['x'].values.astype(np.float64)
    y = df['y'].values.astype(np.float64)
    z = df['z'].values.astype(np.float64)
    n = len(x)
    
    bin_file = os.path.join(OUTPUT_DIR, f"sat_{sat_id}.bin")
    with open(bin_file, 'wb') as f:
        # Write n_points as 4-byte int (matching C++ sizeof(int))
        f.write(struct.pack('i', n))
        # Write x, y, z only (3 doubles per point)
        for i in range(n):
            f.write(struct.pack('ddd', x[i], y[i], z[i]))
    
    if sat_id % 100 == 0:
        print(f"  [{sat_id}/1000] sat_{sat_id}.bin: {n} points, layer={sat['layer']}, height={sat['height']}km")

# Verify
test_file = os.path.join(OUTPUT_DIR, "sat_1.bin")
fsize = os.path.getsize(test_file)
print(f"\nDone! orbit_bin/ created.")
print(f"  sat_1.bin size: {fsize:,} bytes")
print(f"  Expected: 4 + {259201}*24 = {4+259201*24:,} bytes")