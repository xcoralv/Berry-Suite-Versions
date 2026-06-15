# compression_utils.py
import bz2
import time
import numpy as np
import multiprocessing as mp
import ctypes

# ===============================
# Mantissa truncation (unchanged)
# ===============================
def trunc_mantissa_float64_array(arr, keep_bits=48):
    a = np.ascontiguousarray(arr, dtype=np.float64)
    bits = a.view(np.uint64)

    mantissa_mask = np.uint64((1 << 52) - 1)
    exponent_and_sign = bits & (~mantissa_mask)
    mantissa = bits & mantissa_mask

    shift = 52 - keep_bits
    mantissa = (mantissa >> shift) << shift

    return (exponent_and_sign | mantissa).view(np.float64).reshape(arr.shape)

def truncate_complex(arr, keep_bits=24):
    return (
        trunc_mantissa_float64_array(arr.real, keep_bits)
        + 1j * trunc_mantissa_float64_array(arr.imag, keep_bits)
    )

# ===============================
# Compression / Decompression
# ===============================
def save_bz2_npy(filename, arr):
    """Save numpy array to bz2 file and return compression time."""
    t0 = time.time()
    with bz2.open(filename, "wb") as f:
        np.save(f, arr)
    dt = time.time() - t0
    return dt

def load_bz2_npy(filename):
    """Load numpy array from bz2 file and return array + decompression time."""
    t0 = time.time()
    with bz2.open(filename, "rb") as f:
        arr = np.load(f)
    dt = time.time() - t0
    return arr, dt

# ===============================
# Aggregate timing reporting
# ===============================
class CompressionTimer:
    def __init__(self):
        self.compress_times = []
        self.decompress_times = []

    def add_compression(self, dt):
        self.compress_times.append(dt)

    def add_decompression(self, dt):
        self.decompress_times.append(dt)

    def report(self):
        total_comp = sum(self.compress_times)
        total_decomp = sum(self.decompress_times)
        print("\n================ Compression timings ================")
        print(f"COMPRESSION TIME: {total_comp:.6f} s")
        print(f"DECOMPRESSION TIME: {total_decomp:.6f} s")
        print("=====================================================\n")
        return total_comp, total_decomp
