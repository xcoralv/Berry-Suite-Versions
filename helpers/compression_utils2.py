# compression_utils.py
import os
import time
import numpy as np
import bz2
import lzma
import gzip
import zlib
from io import BytesIO

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

# ============================================================
# Mantissa truncation (unchanged)
# ============================================================
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

# ============================================================
# Compression method selection
# ============================================================
COMPRESSION_METHOD = os.environ.get("BERRY_COMPRESSION", "bz2").lower()

# ============================================================
# Helpers
# ============================================================
def _array_to_npy_bytes(arr):
    buf = BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

def _npy_bytes_to_array(data):
    return np.load(BytesIO(data))

# ============================================================
# Save/load wrappers
# ============================================================
def save_array(filename, arr):
    """Save numpy array and return compression time."""
    t0 = time.time()

    if COMPRESSION_METHOD == "bz2":
        with bz2.open(filename, "wb") as f:
            np.save(f, arr)

    elif COMPRESSION_METHOD == "lzma":
        with lzma.open(filename, "wb") as f:
            np.save(f, arr)

    elif COMPRESSION_METHOD == "gzip":
        with gzip.open(filename, "wb") as f:
            np.save(f, arr)

    elif COMPRESSION_METHOD == "zlib":
        raw = _array_to_npy_bytes(arr)
        compressed = zlib.compress(raw)
        with open(filename, "wb") as f:
            f.write(compressed)

    elif COMPRESSION_METHOD == "npz":
        np.savez_compressed(filename, arr=arr)

    elif COMPRESSION_METHOD == "zstd":
        if not ZSTD_AVAILABLE:
            raise ValueError("Zstandard not available")
        raw = _array_to_npy_bytes(arr)
        cctx = zstd.ZstdCompressor()
        compressed = cctx.compress(raw)
        with open(filename, "wb") as f:
            f.write(compressed)

    else:
        raise ValueError(f"Unknown compression method: {COMPRESSION_METHOD}")

    return time.time() - t0


def load_array(filename):
    """Load numpy array and return (array, decompression time)."""
    t0 = time.time()

    if COMPRESSION_METHOD == "bz2":
        with bz2.open(filename, "rb") as f:
            arr = np.load(f)

    elif COMPRESSION_METHOD == "lzma":
        with lzma.open(filename, "rb") as f:
            arr = np.load(f)

    elif COMPRESSION_METHOD == "gzip":
        with gzip.open(filename, "rb") as f:
            arr = np.load(f)

    elif COMPRESSION_METHOD == "zlib":
        with open(filename, "rb") as f:
            compressed = f.read()
        raw = zlib.decompress(compressed)
        arr = _npy_bytes_to_array(raw)

    elif COMPRESSION_METHOD == "npz":
        arr = np.load(filename)["arr"]

    elif COMPRESSION_METHOD == "zstd":
        if not ZSTD_AVAILABLE:
            raise ValueError("Zstandard not available")
        with open(filename, "rb") as f:
            compressed = f.read()
        dctx = zstd.ZstdDecompressor()
        raw = dctx.decompress(compressed)
        arr = _npy_bytes_to_array(raw)

    else:
        raise ValueError(f"Unknown compression method: {COMPRESSION_METHOD}")

    return arr, time.time() - t0


# ============================================================
# Timing aggregator
# ============================================================
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
        print(f"COMPRESSION TIME:   {total_comp:.6f} s")
        print(f"DECOMPRESSION TIME: {total_decomp:.6f} s")
        print("=====================================================\n")
        return total_comp, total_decomp
