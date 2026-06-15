"""
This program finds the problematic cases and makes a basis rotation of
the wavefunctions
"""
import os
import logging
from scipy.optimize import minimize
import numpy as np
from berry import log
from berry._subroutines.write_k_points import _bands_numbers
from berry._subroutines.clustering_libs import evaluate_result
import sys

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
    from berry.compression_utils1 import load_bz2_npy, CompressionTimer
    from berry.compression_utils2 import save_array, load_array
except:
    pass

timer = CompressionTimer()
# Import bz2 loader
#from compression_utils import load_bz2_npy

CORRECT = 5
POTENTIAL_CORRECT = 4
POTENTIAL_MISTAKE = 3
DEGENERATE = 2
MISTAKE = 1
NOT_SOLVED = 0

def func(aa, ddot):
    r1 = complex(ddot[0], ddot[1])*aa[0]*np.exp(1j*aa[1]) + complex(ddot[2], ddot[3])*np.sqrt(1 - aa[0]**2)*np.exp(1j*aa[2])
    r2 = complex(ddot[4], ddot[5])*aa[3]*np.exp(1j*aa[4]) + complex(ddot[6], ddot[7])*np.sqrt(1 - aa[3]**2)*np.exp(1j*aa[5])
    return -np.absolute(r1) - np.absolute(r2)
    
def set_new_signal(k, bn, psinew, bnfinal, sigfinal, connections, logger: log):
    machbn = bnfinal[k, bn]

    dot_products = []
    for i_neig, kneig in enumerate(d.neighbors[k]):
        if kneig == -1:
            continue

        bneig = bnfinal[kneig, bn]
        psineig, dt = load_array(os.path.join(m.wfcdirectory, f"k0{kneig}b0{bneig}.wfc"))
        timer.add_decompression(dt)

        dphase = d_phase[:, k] * np.conjugate(d_phase[:, kneig])
        dot_product = np.sum(dphase * psinew * np.conjugate(psineig)) / m.nr
        dp = np.abs(dot_product)
        logger.info(f'\told_dp: {connections[k, i_neig, machbn, bneig]} new_dp: {dp}')
        dot_products.append(dp)

        dot_products_neigs = []
        for j_neig, k2_neig in enumerate(d.neighbors[kneig]):
            if k2_neig == -1:
                continue
            bn2neig = bnfinal[k2_neig, bn]
            connection = dp if k2_neig == k else connections[kneig, j_neig, bneig, bn2neig]
            dot_products_neigs.append(connection)
        new_signal = evaluate_result(dot_products_neigs)
        logger.info(f'\told_signal: {sigfinal[kneig, bn]} new_signal: {new_signal}')
        sigfinal[kneig, bn] = new_signal
    
    sigfinal[k, bn] = evaluate_result(dot_products)
    return sigfinal

def run_basis_rotation(max_band: int, npr: int = 1, logger_name: str = "basis",
                       logger_level: int = logging.INFO, compress: bool = False, flush: bool = False):
    global signalfinal, d_phase
    logger = log(logger_name, "BASIS ROTATION", level=logger_level, flush=flush)
    logger.header()

    logger.info("\tUnique reference of run:", m.refname)
    logger.info("\tDirectory where the wfc are:", m.wfcdirectory)
    logger.info("\tNumber of k-points in each direction:", m.nkx, m.nky, m.nkz)
    logger.info("\tTotal number of k-points:", m.nks)
    logger.info("\tTotal number of points in real space:", m.nr)
    logger.info("\tNumber of bands:", m.nbnd)
    logger.info()

    if m.noncolin:
        logger.info("\n\tThis is a noncolinear calculation: basis rotation is not implemented.")
        logger.info("\tExiting.")
        logger.footer()
        exit(0)

    d_phase = np.load(os.path.join(m.data_dir, "phase.npy"))
    logger.info("\tPhases loaded")

    dotproduct = np.load(os.path.join(m.data_dir, "dpc.npy"))
    connections = np.load(os.path.join(m.data_dir, "dp.npy"))
    logger.info("\tDot product loaded")

    logger.info("\tReading files bandsfinal.npy and signalfinal.npy")
    bandsfinal = np.load(os.path.join(m.data_dir, "bandsfinal.npy"))
    signalfinal = np.load(os.path.join(m.data_dir, "signalfinal.npy"))
    degeneratefinal = np.load(os.path.join(m.data_dir, "degeneratefinal.npy"))

    logger.info("\tIdentifying states to apply rotation")
    if degeneratefinal.shape[0] == 0:
        logger.footer()
        exit(0)

    # Filter bands below max_band and only DEGENERATE
    kpproblem = degeneratefinal[:, 0]
    bnproblem = degeneratefinal[:, [1, 2]]
    S1 = signalfinal[kpproblem, bnproblem[:, 0]]
    S2 = signalfinal[kpproblem, bnproblem[:, 1]]
    bands_use = np.logical_and(np.logical_and(bnproblem[:, 0] <= max_band, bnproblem[:, 1] <= max_band),
                               np.logical_and(S1 == DEGENERATE, S2 == DEGENERATE))
    if np.sum(bands_use) == 0:
        logger.footer()
        exit(0)

    kpproblem = kpproblem[bands_use]
    bnproblem = bnproblem[bands_use]
    machbandproblem = np.array(list(zip(bandsfinal[kpproblem, bnproblem[:, 0]],
                                        bandsfinal[kpproblem, bnproblem[:, 1]])))

    # -----------------------------
    # NEW: accumulate decompression locally
    # -----------------------------
    total_decomp_time = 0.0

    for nki, nk0 in enumerate(kpproblem):
        logger.info(f"\tK-point where problem will be solved: {nk0}")
        for j in range(4):  # Find neighbors
            nk = d.neighbors[nk0, j]
            if nk != -1 and signalfinal[nk,bnproblem[nki, 0]] > DEGENERATE and signalfinal[nk,bnproblem[nki, 1]] > DEGENERATE:
                nb1 = machbandproblem[nki, 0]
                nb2 = machbandproblem[nki, 1]
                nkj = j
                break

        dotA1 = dotproduct[nk0, nkj, nb1, nb1]
        dotA2 = dotproduct[nk0, nkj, nb1, nb2]
        dotB1 = dotproduct[nk0, nkj, nb2, nb1]
        dotB2 = dotproduct[nk0, nkj, nb2, nb2]
        dot = np.array([np.real(dotA1), np.imag(dotA1), np.real(dotA2), np.imag(dotA2),
                        np.real(dotB1), np.imag(dotB1), np.real(dotB2), np.imag(dotB2)])

        a = np.array([0.5, 0, 0, 0.5, 0, 0])
        const = ({'type': 'eq', 'fun': lambda a: a[0]*a[3]*np.cos(a[4] - a[1]) + np.sqrt(1 - a[0]**2)*np.sqrt(1 - a[3]**2)*np.cos(a[5] - a[2])},
                 {'type': 'eq', 'fun': lambda a: a[0]*a[3]*np.sin(a[4] - a[1]) + np.sqrt(1 - a[0]**2)*np.sqrt(1 - a[3]**2)*np.sin(a[5] - a[2])})

        from scipy.optimize import minimize
        res = minimize(func, a, args=dot, bounds=[(-1,1),(-np.pi,np.pi),(-np.pi,np.pi),(-1,1),(-np.pi,np.pi),(-np.pi,np.pi)],
                       options={'disp':False}, constraints=const)

        ca1 = res.x[0]*np.exp(1j*res.x[1])
        ca2 = np.sqrt(1 - res.x[0]**2)*np.exp(1j*res.x[2])
        cb1 = res.x[3]*np.exp(1j*res.x[4])
        cb2 = np.sqrt(1 - res.x[3]**2)*np.exp(1j*res.x[5])

        psi1, dt = load_array(os.path.join(d.wfcdirectory, f"k0{nk0}b0{nb1}.wfc"))
        total_decomp_time += dt
        psi2, dt = load_array(os.path.join(d.wfcdirectory, f"k0{nk0}b0{nb2}.wfc"))
        total_decomp_time += dt

        psinewA = psi1*ca1 + psi2*ca2
        psinewB = psi1*cb1 + psi2*cb2

        signalfinal = set_new_signal(nk0, nb1, psinewA, bandsfinal, signalfinal, connections, logger)
        signalfinal = set_new_signal(nk0, nb2, psinewB, bandsfinal, signalfinal, connections, logger)

        # Save new files
        outfileA = os.path.join(d.wfcdirectory, f"k0{nk0}b0{nb1}.wfc1")
        outfileB = os.path.join(d.wfcdirectory, f"k0{nk0}b0{nb2}.wfc1")
        with open(outfileA, "wb") as f: np.save(f, psinewA)
        with open(outfileB, "wb") as f: np.save(f, psinewB)

    # Add total decompression to timer once
    timer.add_decompression(total_decomp_time)
    print('Total decompression time:', total_decomp_time)

    np.save(os.path.join(m.data_dir, 'signalfinal.npy'), signalfinal)
    logger.info(f"\tTotal decompression time: {total_decomp_time:.4f}s")
    logger.footer()

if __name__ == "__main__":
    #run_basis_rotation(9)
    run_basis_rotation(9, log("basisrotation", "BASIS ROTATION", "version", logging.DEBUG))
    timer.report()
