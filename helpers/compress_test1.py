#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np

# ------------------------------
# Utilities
# ------------------------------

def l2_rel_error(a, b):
    """Relative L2 norm error."""
    return np.linalg.norm(a - b) / np.linalg.norm(a)

def overlap_error(a, b):
    """Deviation from perfect overlap (normalized inner product)."""
    num = np.vdot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1 - abs(num) / denom if denom != 0 else np.nan

# ------------------------------
# Compression tricks
# ------------------------------

def downcast_float(arr, dtype):
    return arr.astype(dtype)

def threshold_array(arr, thresh, rel=False):
    if rel:
        norm = np.max(np.abs(arr))
        thresh = thresh * norm
    out = arr.copy()
    out[np.abs(out) < thresh] = 0.0
    return out

def trunc_mantissa_float64_array(arr, keep_bits):
    """Keep only the top `keep_bits` mantissa bits in a float64 array."""
    a = np.ascontiguousarray(arr, dtype=np.float64)
    bits = a.view(np.uint64)
    mantissa_mask = np.uint64((1 << 52) - 1)
    exponent_and_sign = bits & (~mantissa_mask)
    mantissa = bits & mantissa_mask
    shift = 52 - keep_bits
    mantissa = (mantissa >> shift) << shift
    new_bits = exponent_and_sign | mantissa
    return new_bits.view(np.float64).reshape(arr.shape)

# ------------------------------
# Load QE wavefunctions
# ------------------------------

def load_wfc_grouped(directory):
    """
    Load all kXXbYY.wfc files and group them by k-point.
    Returns:
        all_k_orig: list of (nCoeffs, nBands) arrays, one per k-point
        k_indices: list of k-point indices
        b_indices: list of band indices
    """
    files = [f for f in os.listdir(directory) if f.endswith(".wfc")]
    if not files:
        raise ValueError(f"No .wfc files found in {directory}")

    pattern = re.compile(r"k0?(\d+)b0?(\d+)\.wfc")
    entries = []
    for fname in files:
        match = pattern.match(fname)
        if not match:
            continue
        k_idx, b_idx = map(int, match.groups())
        psi = np.load(os.path.join(directory, fname))
        entries.append(((k_idx, b_idx), psi))

    # Sort by (k, band)
    entries.sort(key=lambda x: (x[0][0], x[0][1]))

    ks = sorted(set(k for (k, b), _ in entries))
    bs = sorted(set(b for (k, b), _ in entries))

    all_k_orig = []
    for k in ks:
        band_arrays = []
        for b in bs:
            psi = [arr for ((kk, bb), arr) in entries if kk == k and bb == b]
            if psi:
                band_arrays.append(psi[0])
        psi_k = np.stack(band_arrays, axis=1)  # shape (nCoeffs, nBands)
        all_k_orig.append(psi_k)

    return all_k_orig, ks, bs

# ------------------------------
# Continuity check (updated)
# ------------------------------

def continuity_error_same_band(all_k_orig, all_k_comp):
    """
    Compute continuity deviation between neighboring k-points,
    comparing only the same band. Guarantees deviation in [0,1].
    """
    deviations = []

    for k in range(len(all_k_orig) - 1):
        psi_k_orig = all_k_orig[k]
        psi_k1_orig = all_k_orig[k + 1]
        psi_k_comp = all_k_comp[k]
        psi_k1_comp = all_k_comp[k + 1]

        nBands = psi_k_orig.shape[1]
        dev_per_band = []

        for i in range(nBands):
            # Normalize each band
            orig_overlap = np.vdot(psi_k_orig[:, i], psi_k1_orig[:, i])
            orig_overlap /= (np.linalg.norm(psi_k_orig[:, i]) * np.linalg.norm(psi_k1_orig[:, i]))

            comp_overlap = np.vdot(psi_k_comp[:, i], psi_k1_comp[:, i])
            comp_overlap /= (np.linalg.norm(psi_k_comp[:, i]) * np.linalg.norm(psi_k1_comp[:, i]))

            # Absolute difference of magnitudes
            dev_per_band.append(abs(abs(orig_overlap) - abs(comp_overlap)))

        deviations.append(max(dev_per_band))

    return deviations

# ------------------------------
# Compression experiment runner
# ------------------------------

def run_experiments(psi, threshold=None, rel_threshold=None, truncate_bits=None):
    results = {}

    # Downcasts
    results["float32_downcast"] = downcast_float(psi, np.complex64)
    results["float16_downcast"] = psi.real.astype(np.float16) + 1j*psi.imag.astype(np.float16)

    # Thresholding
    if threshold is not None:
        results[f"abs_thresh_{threshold:.1e}"] = threshold_array(psi, threshold, rel=False)
    if rel_threshold is not None:
        results[f"rel_thresh_{rel_threshold:.1e}"] = threshold_array(psi, rel_threshold, rel=True)

    # Mantissa truncation
    if truncate_bits:
        for kb in truncate_bits:
            real_trunc = trunc_mantissa_float64_array(psi.real, kb)
            imag_trunc = trunc_mantissa_float64_array(psi.imag, kb)
            results[f"trunc_{kb}bits"] = real_trunc + 1j*imag_trunc

    return results

# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compression + continuity test for QE wavefunctions")
    parser.add_argument("wfc_dir", help="Directory containing kXXbYY.wfc files")
    parser.add_argument("--truncate-bits", type=int, nargs="*", default=[], help="Mantissa bit truncation levels to test")
    parser.add_argument("--threshold", type=float, default=None, help="Absolute threshold for small coefficients")
    parser.add_argument("--rel-threshold", type=float, default=None, help="Relative threshold for small coefficients")
    parser.add_argument("--out", default=None, help="Directory to save compressed wavefunctions")
    args = parser.parse_args()

    print(f"Loading .wfc files from {args.wfc_dir}...")
    all_k_orig, ks, bs = load_wfc_grouped(args.wfc_dir)
    print(f"Loaded {len(ks)} k-points, {len(bs)} bands, {all_k_orig[0].shape[0]} coefficients each.")

    # Flatten for system-level compression
    psi_flat = np.concatenate([psi_k.ravel() for psi_k in all_k_orig])

    results = run_experiments(psi_flat, threshold=args.threshold,
                              rel_threshold=args.rel_threshold,
                              truncate_bits=args.truncate_bits)

    print("\n=== Global Compression Results ===")
    for name, arr_flat in results.items():
        # Compute metrics
        l2 = l2_rel_error(psi_flat, arr_flat)
        overlap_dev = overlap_error(psi_flat, arr_flat)
        raw_bytes = arr_flat.nbytes
        overlap_str = f"{overlap_dev:.3e}" if overlap_dev is not None else "N/A"
        print(f"{name:25s} | L2_rel: {l2:.3e} | overlap_dev: {overlap_str:>10} | raw: {raw_bytes:10d}")

        # Rebuild compressed wavefunctions per k-point
        all_k_comp = []
        idx = 0
        for psi_k in all_k_orig:
            nCoeffs, nBands = psi_k.shape
            size = nCoeffs * nBands
            psi_k_flat = arr_flat[idx:idx+size].reshape(nCoeffs, nBands)
            idx += size
            all_k_comp.append(psi_k_flat)

        # Compute continuity deviations
        cont_dev = continuity_error_same_band(all_k_orig, all_k_comp)
        print(f"    Continuity deviation per neighbor k: {[f'{d:.2e}' for d in cont_dev]}")
        print(f"    Max continuity deviation: {max(cont_dev):.2e}")

        # Optionally save results
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            np.save(os.path.join(args.out, f"{name}.npy"), arr_flat)

if __name__ == "__main__":
    main()
