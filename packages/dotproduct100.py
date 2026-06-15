from multiprocessing import Pool, Array
from typing import Tuple, Optional
import sys
import os
from time import time
import ctypes
import logging
import numpy as np

from berry import log

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
    # Ensure sys.path includes berry folder if needed
    sys.path.insert(0, str(os.path.expanduser("~/berry-Version-2.0/berry")))
    from compression_utils1 import load_bz2_npy, CompressionTimer
    from berry.compression_utils2 import save_array, load_array
except Exception as e:
    print("Error importing modules:", e)
    raise

timer = CompressionTimer()  # main process timer

# ------------------------------------------------------------------------
# Dot product function for Pool workers
# ------------------------------------------------------------------------
def dot(nk: int, j: int, neighbor: int, jNeighbor: Tuple[np.ndarray]) -> float:
    """
    Calculate dot products for one k-point pair and return total decompression time.
    """
    local_decomp_time = 0.0

    dphase = d_phase[:, nk] * d_phase[:, neighbor].conj()

    if m.noncolin:
        # Non-colinear case (add decompression timing similarly if used)
        for band0 in range(m.nbnd):
            wfc00, dt = load_bz2_npy(os.path.join(m.wfcdirectory, f"k0{nk}b0{band0}-0.wfc.bz2"))
            wfc01, dt = load_bz2_npy(os.path.join(m.wfcdirectory, f"k0{nk}b0{band0}-1.wfc.bz2"))
            local_decomp_time += dt * 2

            for band1 in range(m.nbnd):
                wfc10, dt = load_array(os.path.join(m.wfcdirectory, f"k0{neighbor}b0{band1}-0.wfc"))
                wfc11, dt = load_array(os.path.join(m.wfcdirectory, f"k0{neighbor}b0{band1}-1.wfc"))
                local_decomp_time += dt * 2

                wfc10 = wfc10.conj()
                wfc11 = wfc11.conj()

                dpc[nk, j, band0, band1] = np.einsum("k,k,k->", dphase, wfc00, wfc10) \
                                            + np.einsum("k,k,k->", dphase, wfc01, wfc11)
                dpc[neighbor, jNeighbor, band1, band0] = dpc[nk, j, band0, band1].conj()
    else:
        # Non-relativistic case
        for band0 in range(m.nbnd):
            wfc0, dt = load_array(os.path.join(m.wfcdirectory, f"k0{nk}b0{band0}.wfc"))
            local_decomp_time += dt
            for band1 in range(m.nbnd):
                wfc1, dt = load_array(os.path.join(m.wfcdirectory, f"k0{neighbor}b0{band1}.wfc"))
                local_decomp_time += dt
                wfc1 = wfc1.conj()
                dpc[nk, j, band0, band1] = np.einsum("k,k,k->", dphase, wfc0, wfc1)
                dpc[neighbor, jNeighbor, band1, band0] = dpc[nk, j, band0, band1].conj()

    return local_decomp_time

# ------------------------------------------------------------------------
# Generate neighbor arguments
# ------------------------------------------------------------------------
def get_point_neighbors(nk: int, j: int) -> Optional[Tuple[int, int, int, Tuple[np.ndarray]]]:
    neighbor = d.neighbors[nk, j]
    if neighbor != -1 and neighbor > nk:
        jNeighbor = np.where(d.neighbors[neighbor] == nk)
        return (nk, j, neighbor, jNeighbor)
    return None

# ------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------
def run_dot(npr: Optional[int] = None, logger_name: str = "dot", logger_level: int = logging.INFO, flush: bool = False):
    global dpc, d_phase, logger
    logger = log(logger_name, "DOT PRODUCT", level=logger_level, flush=flush)
    logger.header()

    if not 0 < npr <= os.cpu_count():
        raise ValueError(f"npr must be between 1 and {os.cpu_count()}")
    logger.info(f"\tUsing {npr} processes")

    # ------------------------------------------------------------------------
    # 1. DEFINE SHARED ARRAYS
    # ------------------------------------------------------------------------
    DPC_SIZE = m.nks * 2 * m.dimensions * m.nbnd * m.nbnd
    DPC_SHAPE = (m.nks, 2 * m.dimensions, m.nbnd, m.nbnd)

    dpc_base = Array(ctypes.c_double, 2 * DPC_SIZE, lock=False)
    dpc = np.frombuffer(dpc_base, dtype=np.complex128).reshape(DPC_SHAPE)
    dp = np.zeros(DPC_SHAPE, dtype=np.float64)
    d_phase = np.load(os.path.join(m.workdir, os.path.join(m.data_dir, "phase.npy")))

    # ------------------------------------------------------------------------
    # 2. GENERATE ARGUMENTS
    # ------------------------------------------------------------------------
    pre_connection_args = [
        args
        for nk in range(m.nks)
        for j in range(2 * m.dimensions)
        if (args := get_point_neighbors(nk, j)) is not None
    ]

    # ------------------------------------------------------------------------
    # 3. CALCULATE DOT PRODUCTS WITH POOL
    # ------------------------------------------------------------------------
    start_total = time()
    total_decomp_time = 0.0
    with Pool(npr) as pool:
        results = pool.starmap(dot, pre_connection_args)
        total_decomp_time = sum(results)
    elapsed_total = time() - start_total

    timer.add_decompression(total_decomp_time)
    print('total decomp time:', total_decomp_time)

    # ------------------------------------------------------------------------
    # 4. POSTPROCESS AND SAVE
    # ------------------------------------------------------------------------
    dpc /= m.nr
    dp = np.abs(dpc)

    np.save(os.path.join(m.data_dir, "dpc.npy"), dpc)
    np.save(os.path.join(m.data_dir, "dp.npy"), dp)
    logger.info(f"\n\tDot products saved to file dpc.npy")
    logger.info(f"\tDot products modulus saved to file dp.npy")
    logger.info(f"\tTotal elapsed time: {elapsed_total:.2f}s")
    logger.info(f"\tTotal decompression time: {total_decomp_time:.4f}s")

    logger.footer()

# ------------------------------------------------------------------------
if __name__ == "__main__":
    run_dot(npr=20)
    #timer.report()  # prints COMPRESSION / DECOMPRESSION totals
