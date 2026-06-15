from multiprocessing import Pool, Array
from typing import Literal

import os
from time import time
import ctypes
import logging

import numpy as np

from berry import log
from berry.utils.jit import numba_njit
from berry.shg import load_berry_connections

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
    from berry.compression_utils1 import load_bz2_npy, CompressionTimer
    from berry.compression_utils2 import save_array, load_array
except:
    pass

timer = CompressionTimer()
total_decomp_time = 0.0
total_decomp_time_parallel = 0.0

def deriv(berryConnection, s, sprime, alpha1, alpha2, dk):
    from findiff import Gradient
    if m.dimensions == 1:
        grad = Gradient(h=[dk], acc=4)
    elif m.dimensions == 2:
        grad = Gradient(h=[dk, dk], acc=4)
    else:
        grad = Gradient(h=[dk, dk, dk], acc=4)

    a = grad(berryConnection[s][sprime][alpha1])
    return a[alpha2]


def berry_connection(n_pos: int, n_gra: int):
    td = 0
    if m.noncolin:
        wfcpos0, dt0 = load_bz2_npy(os.path.join(m.data_dir, f"wfcpos{n_pos}-0.npy.bz2"))
        wfcpos1, dt1 = load_bz2_npy(os.path.join(m.data_dir, f"wfcpos{n_pos}-1.npy.bz2"))
        timer.add_decompression(dt0 + dt1)
        td += dt0
        td += dt1
        wfcpos0 = wfcpos0.conj()
        wfcpos1 = wfcpos1.conj()

        @numba_njit
        def aux_connection():
            bcc = np.zeros(wfcgra0[0].shape, dtype=np.complex128)
            for posi in range(m.nr):
                bcc += 1j * (wfcpos0[posi] * wfcgra0[posi] +
                             wfcpos1[posi] * wfcgra1[posi])
            return bcc / m.nr
    else:
        wfcpos, dt = load_array(os.path.join(m.data_dir, f"wfcpos{n_pos}.npy"))
        timer.add_decompression(dt)
        td += dt
        wfcpos = wfcpos.conj()

        @numba_njit
        def aux_connection():
            bcc = np.zeros(wfcgra[0].shape, dtype=np.complex128)
            for posi in range(m.nr):
                bcc += 1j * wfcpos[posi] * wfcgra[posi]
            return bcc / m.nr

    start = time()
    bcc = aux_connection()
    logger.info(f"\tberry_connection{n_pos}_{n_gra} calculated in {time() - start:.2f} seconds")
    np.save(os.path.join(m.geometry_dir, f"berryConn{n_pos}_{n_gra}.npy"), bcc)
    return td

def chern_number(curv) -> np.complex128:
    chern = 0
    if m.dimensions == 2:
        chern = np.sum(curv) * (np.linalg.norm(m.b1) / m.nkx) * (np.linalg.norm(m.b2) / m.nky) / (2 * np.pi)
    else:  # 3D 
        chern = (np.sum(curv[0]) * np.linalg.norm(m.b1) / m.nkx
               + np.sum(curv[1]) * np.linalg.norm(m.b2) / m.nky
               + np.sum(curv[2]) * np.linalg.norm(m.b3) / m.nkz) / (2 * np.pi)

    return chern

def berry_phase(pos, x, y):
    bp = (np.sum(pos[:, x, y].conj()     * pos[:, x+1, y]) 
       *  np.sum(pos[:, x+1, y].conj()   * pos[:, x+1, y+1])
       *  np.sum(pos[:, x+1, y+1].conj() * pos[:, x, y+1])
       *  np.sum(pos[:, x, y+1].conj()   * pos[:, x, y])) / (m.nr ** 4)

    bp = np.angle(bp)

    return bp

def chern_number_bp(pos) -> np.complex128:
    chern = 0
    if m.dimensions == 2:
        for i in range(m.nkx -1):
            for j in range(m.nky -1):
                chern += berry_phase(pos, i, j)
        chern /= (2*np.pi)
#    else:  # 3D 
#        chern = (np.sum(curv[0]) * np.linalg.norm(m.b1) / m.nkx
#               + np.sum(curv[1]) * np.linalg.norm(m.b2) / m.nky
#               + np.sum(curv[2]) * np.linalg.norm(m.b3) / m.nkz) / (2 * np.pi)

    return chern

def chern_number_bp_bz(pos) -> np.complex128:
    chern = 1
    if m.dimensions == 2:
        for i in range(m.nkx-1):
            chern *= np.complex256(np.sum(pos[:, i, 0].conj() * pos[:, i+1, 0])) * np.complex256(np.sum(pos[:, i, m.nky-1] * pos[:, i+1, m.nky-1].conj()))
        for j in range(m.nky-1):
            chern *= np.complex256(np.sum(pos[:, m.nkx-1, j].conj() * pos[:, m.nkx-1, j+1])) * np.complex256(np.sum(pos[:, 0, j] * pos[:, 0, j+1].conj()))
        chern = np.angle(chern)
        chern /= (2*np.pi)
#    else:  # 3D 
#        chern = (np.sum(curv[0]) * np.linalg.norm(m.b1) / m.nkx
#               + np.sum(curv[1]) * np.linalg.norm(m.b2) / m.nky
#               + np.sum(curv[2]) * np.linalg.norm(m.b3) / m.nkz) / (2 * np.pi)

    return chern




def berry_curvature(idx: int, idx_: int):
    td = 0
    if m.noncolin:
        if idx == idx_:
            wfcgra0_ = wfcgra0.conj()
            wfcgra1_ = wfcgra1.conj()
        else:
            wfcgra0_, dt0 = load_array(os.path.join(m.data_dir, f"wfcgra{idx_}-0.npy"))
            wfcgra1_, dt1 = load_array(os.path.join(m.data_dir, f"wfcgra{idx_}-1.npy"))
            timer.add_decompression(dt0 + dt1)
            wfcgra0_ = wfcgra0_.conj()
            wfcgra1_ = wfcgra1_.conj()

        if m.dimensions == 2:
            @numba_njit
            def aux_curvature():
                bcr = np.zeros(wfcgra0[0].shape, dtype=np.complex128)
                for posi in range(m.nr):
                    bcr += (
                        1j * wfcgra0[posi][1] * wfcgra0_[posi][0]
                        - 1j * wfcgra0[posi][0] * wfcgra0_[posi][1]
                        + 1j * wfcgra1[posi][1] * wfcgra1_[posi][0]
                        - 1j * wfcgra1[posi][0] * wfcgra1_[posi][1]
                    )
                return bcr / m.nr
        else:
            @numba_njit
            def aux_curvature():
                bcr0 = np.zeros(wfcgra0[0].shape, dtype=np.complex128)
                bcr1 = np.zeros(wfcgra0[0].shape, dtype=np.complex128)
                bcr2 = np.zeros(wfcgra0[0].shape, dtype=np.complex128)
                for posi in range(m.nr):
                    bcr0 += (
                        1j * wfcgra0[posi][2] * wfcgra0_[posi][1]
                        - 1j * wfcgra0[posi][1] * wfcgra0_[posi][2]
                        + 1j * wfcgra1[posi][2] * wfcgra1_[posi][1]
                        - 1j * wfcgra1[posi][1] * wfcgra1_[posi][2]
                    )
                    bcr1 += (
                        1j * wfcgra0[posi][0] * wfcgra0_[posi][2]
                        - 1j * wfcgra0[posi][2] * wfcgra0_[posi][0]
                        + 1j * wfcgra1[posi][0] * wfcgra1_[posi][2]
                        - 1j * wfcgra1[posi][2] * wfcgra1_[posi][0]
                    )
                    bcr2 += (
                        1j * wfcgra0[posi][1] * wfcgra0_[posi][0]
                        - 1j * wfcgra0[posi][0] * wfcgra0_[posi][1]
                        + 1j * wfcgra1[posi][1] * wfcgra1_[posi][0]
                        - 1j * wfcgra1[posi][0] * wfcgra1_[posi][1]
                    )
                return bcr0 / m.nr, bcr1 / m.nr, bcr2 / m.nr
    else:
        if idx == idx_:
            wfcgra_ = wfcgra.conj()
        else:
            wfcgra_, dt = load_array(os.path.join(m.data_dir, f"wfcgra{idx_}.npy"))
            timer.add_decompression(dt)
            td += dt
            wfcgra_ = wfcgra_.conj()

        if m.dimensions == 2:
            @numba_njit
            def aux_curvature():
                bcr = np.zeros(wfcgra[0].shape, dtype=np.complex128)
                for posi in range(m.nr):
                    bcr += (
                        1j * wfcgra[posi][0] * wfcgra_[posi][1]
                        - 1j * wfcgra[posi][1] * wfcgra_[posi][0]
                    )
                return bcr / m.nr
        else:
            @numba_njit
            def aux_curvature():
                bcr0 = np.zeros(wfcgra[0].shape, dtype=np.complex128)
                bcr1 = np.zeros(wfcgra[0].shape, dtype=np.complex128)
                bcr2 = np.zeros(wfcgra[0].shape, dtype=np.complex128)
                for posi in range(m.nr):
                    bcr0 += (
                        1j * wfcgra[posi][2] * wfcgra_[posi][1]
                        - 1j * wfcgra[posi][1] * wfcgra_[posi][2]
                    )
                    bcr1 += (
                        1j * wfcgra[posi][0] * wfcgra_[posi][2]
                        - 1j * wfcgra[posi][2] * wfcgra_[posi][0]
                    )
                    bcr2 += (
                        1j * wfcgra[posi][1] * wfcgra_[posi][0]
                        - 1j * wfcgra[posi][0] * wfcgra_[posi][1]
                    )
                return bcr0 / m.nr, bcr1 / m.nr, bcr2 / m.nr

    start = time()
    bcr = aux_curvature() if m.dimensions == 2 else np.array(aux_curvature())
    logger.info(f"\tberry_curvature{idx}_{idx_} calculated in {time() - start:.2f} seconds")
    np.save(os.path.join(m.geometry_dir, f"berryCur{idx}_{idx_}.npy"), bcr)
    return td

def berry_curvature_curl(idx: int, idx_: int, berry_connection) -> None:
    """
    Calculates the Berry curvature using the curl of Berry connections.
    """

    
    
    if m.dimensions == 2:                # 2D case
#        @numba_njit
        def aux_curvature() -> np.ndarray:
            """
            Auxiliary function to calculate the Berry curvature.
            Attention: this is valid for 2D and 3D materials.
            """

            #bcr = np.zeros(wfcgra[0,0].shape, dtype=np.complex128)
            bcr = deriv(berry_connection, idx, idx_, 0, 1, m.step) 
            - deriv(berry_connection, idx, idx_, 1, 0, m.step)
        
            return bcr 
        
    else:                                # 3D case
#        @numba_njit
        def aux_curvature():
            """
            Auxiliary function to calculate the Berry curvature.
            Attention: this is valid for 2D and 3D materials.
            """

            bcr0 = deriv(berry_connection, idx, idx_, 1, 2, m.step)
            - deriv(berry_connection, idx, idx_, 2, 1, m.step)

            bcr1 = deriv(berry_connection, idx, idx_, 2, 0, m.step)
            - deriv(berry_connection, idx, idx_, 0, 2, m.step)

            bcr2 = deriv(berry_connection, idx, idx_, 0, 1, m.step)
            - deriv(berry_connection, idx, idx_, 1, 0, m.step)

            return bcr0, bcr1, bcr2

    start = time()

    bcr = aux_curvature() if m.dimensions == 2 else np.array(aux_curvature())
    logger.info(f"\tberry_curvature{idx}_{idx_}_curl calculated in {time() - start:.2f} seconds")

    np.save(os.path.join(m.geometry_dir, f"berryCur{idx}_{idx_}_curl.npy"), bcr)
    
    return bcr


def run_berry_geometry(max_band: int, min_band: int = 0, npr: int = 1,
                       prop: Literal["curv", "conn", "both"] = "both",
                       logger_name="geometry", logger_level=logging.INFO, flush=False):

    if m.noncolin:
        global wfcgra0, wfcgra1, logger
    else:
        global wfcgra, logger, total_decomp_time, total_decomp_time_parallel

    logger = log(logger_name, "BERRY GEOMETRY", level=logger_level, flush=flush)
    logger.header()

    if m.dimensions == 2:
        GRA_SIZE = m.nr * m.dimensions * m.nkx * m.nky
        GRA_SHAPE = (m.nr, m.dimensions, m.nkx, m.nky)
    else:
        GRA_SIZE = m.nr * m.dimensions * m.nkx * m.nky * m.nkz
        GRA_SHAPE = (m.nr, m.dimensions, m.nkx, m.nky, m.nkz)

    if m.noncolin:
        arr0 = Array(ctypes.c_double, 2 * GRA_SIZE, lock=False)
        arr1 = Array(ctypes.c_double, 2 * GRA_SIZE, lock=False)
        wfcgra0 = np.frombuffer(arr0, dtype=np.complex128).reshape(GRA_SHAPE)
        wfcgra1 = np.frombuffer(arr1, dtype=np.complex128).reshape(GRA_SHAPE)
    else:
        arr = Array(ctypes.c_double, 2 * GRA_SIZE, lock=False)
        wfcgra = np.frombuffer(arr, dtype=np.complex128).reshape(GRA_SHAPE)

    if prop in ("both", "conn"):
        for idx in range(min_band, max_band + 1):
            if m.noncolin:
                tmp, dt = load_bz2_npy(os.path.join(m.data_dir, f"wfcgra{idx}-0.npy.bz2"))
                timer.add_decompression(dt)
                total_decomp_time += dt
                wfcgra0[:] = tmp

                tmp, dt = load_bz2_npy(os.path.join(m.data_dir, f"wfcgra{idx}-1.npy.bz2"))
                timer.add_decompression(dt)
                total_decomp_time += dt
                wfcgra1[:] = tmp
            else:
                tmp, dt = load_array(os.path.join(m.data_dir, f"wfcgra{idx}.npy"))
                timer.add_decompression(dt)
                total_decomp_time += dt
                wfcgra[:] = tmp

            work = ((i, idx) for i in range(min_band, max_band + 1))
            with Pool(npr) as pool:
                results = pool.starmap(berry_connection, work)
                total_decomp_time_parallel += sum(results)

    if prop in ("both", "curv"):
        if m.noncolin:
            for idx in range(min_band, max_band + 1):
                wfcgra0 = np.load(os.path.join(m.data_dir, f"wfcgra{idx}-0.npy"))
                wfcgra1 = np.load(os.path.join(m.data_dir, f"wfcgra{idx}-1.npy"))

                work_load = ((idx, idx_) for idx_ in range(min_band, max_band + 1))

                with Pool(npr) as pool:
                    pool.starmap(berry_curvature, work_load)
        else:
            for idx in range(min_band, max_band + 1):
                tmp, dt = load_array(os.path.join(m.data_dir, f"wfcgra{idx}.npy"))
                timer.add_decompression(dt)
                total_decomp_time += dt
                wfcgra[:] = tmp

                work_load = ((idx, idx_) for idx_ in range(min_band, max_band + 1))

                with Pool(npr) as pool:
                    results = pool.starmap(berry_curvature, work_load)
                    total_decomp_time_parallel += sum(results)


    print('total decomp time parallel:', total_decomp_time_parallel)
    total_decomp_time += total_decomp_time_parallel
    print('total decomp time:', total_decomp_time)
    logger.footer()
    


if __name__ == "__main__":
    run_berry_geometry(9, npr=10, prop="both",
                       logger_name="berry_geometry",
                       logger_level=logging.DEBUG)
    timer.report()
