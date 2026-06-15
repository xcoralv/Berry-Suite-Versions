from multiprocessing import Pool, Array, Manager
from itertools import product
from typing import Tuple
#from joblib import Parallel, delayed
from numba import njit, prange

import os
import ctypes
import logging

from findiff import Gradient

import numpy as np

from berry import log
from berry._subroutines.comutator import comute, comutederiv, comute3

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass


def load_berry_connections(conduction_band: int, berry_conn_size: int, berry_conn_shape: Tuple[int], initial_band: int) -> np.ndarray:
    base = Array(ctypes.c_double, berry_conn_size * 2, lock = False)
    berry_connections = np.frombuffer(base, dtype=np.complex128).reshape(berry_conn_shape)

    for i in range(initial_band, conduction_band + 1):
        for j in range(initial_band, conduction_band + 1):
            berry_connections[i - initial_band, j - initial_band] = np.load(os.path.join(m.geometry_dir, f"berryConn{i}_{j}.npy"))

    return berry_connections


def correct_eigenvalues(bandsfinal: np.ndarray) -> np.ndarray:
    kp = 0
    eigenvalues = d.eigenvalues[:, m.initial_band:] # initial band correction
    if m.dimensions == 1:
        eigen_array = np.zeros((m.nkx, number_of_bands))
        for i in range(m.nkx):
            for banda in range(number_of_bands):
                eigen_array[i, banda] = eigenvalues[kp, bandsfinal[kp, banda]]
            kp += 1

    elif m.dimensions == 2:
        eigen_array = np.zeros((m.nkx, m.nky, number_of_bands))
        for j in range(m.nky):
            for i in range(m.nkx):
                for banda in range(number_of_bands):
                    eigen_array[i, j, banda] = eigenvalues[kp, bandsfinal[kp, banda]]
                kp += 1
    else:
        eigen_array = np.zeros((m.nkx, m.nky, m.nkz, number_of_bands))
        for l in range(m.nkz):
            for j in range(m.nky):
                for i in range(m.nkx):
                    for banda in range(number_of_bands):
                        eigen_array[i, j, l, banda] = eigenvalues[kp, bandsfinal[kp, banda]]
                    kp += 1


    return eigen_array


def get_fermi_delta_ea_grad_ea(grad: Gradient, eigen_array: np.ndarray, conduction_band: int, ) -> Tuple[np.ndarray, np.ndarray]:
    if m.dimensions == 1:
        grad_dea = np.zeros((1, m.nkx, conduction_band + 1, conduction_band + 1), dtype=np.complex128)
        delta_ea = np.zeros((m.nkx, conduction_band + 1, conduction_band + 1))
        fermi = np.zeros((m.nkx, conduction_band + 1, conduction_band + 1))

        for s, sprime in product(band_list, repeat=2):
            delta_ea[:, s, sprime] = eigen_array[:, s] - eigen_array[:, sprime]
            grad_dea[:, :, s, sprime] = grad(delta_ea[:, s, sprime])
            if s <= m.vb - initial_band < sprime:
                fermi[:, s, sprime] = 1
            elif sprime <= m.vb - initial_band < s:
                fermi[:, s, sprime] = -1    
    elif m.dimensions == 2:
        grad_dea = np.zeros((2, m.nkx, m.nky, conduction_band + 1, conduction_band + 1), dtype=np.complex128)
        delta_ea = np.zeros((m.nkx, m.nky, conduction_band + 1, conduction_band + 1))
        fermi = np.zeros((m.nkx, m.nky, conduction_band + 1, conduction_band + 1))

        for s, sprime in product(band_list, repeat=2):
            delta_ea[:, :, s, sprime] = eigen_array[:, :, s] - eigen_array[:, :, sprime]
            grad_dea[:, :, :, s, sprime] = grad(delta_ea[:, :, s, sprime])
            if s <= m.vb - initial_band < sprime:
                fermi[:, :, s, sprime] = 1
            elif sprime <= m.vb - initial_band < s:
                fermi[:, :, s, sprime] = -1
    else:
        grad_dea = np.zeros((3, m.nkx, m.nky, m.nkz, conduction_band + 1, conduction_band + 1), dtype=np.complex128)
        delta_ea = np.zeros((m.nkx, m.nky, m.nkz, conduction_band + 1, conduction_band + 1))
        fermi = np.zeros((m.nkx, m.nky, m.nkz, conduction_band + 1, conduction_band + 1))

        for s, sprime in product(band_list, repeat=2):
            delta_ea[:, :, :, s, sprime] = eigen_array[:, :, :, s] - eigen_array[:, :, :, sprime]
            grad_dea[:, :, :, :, s, sprime] = grad(delta_ea[:, :, :, s, sprime])
            if s <= m.vb - initial_band < sprime:
                fermi[:, :, :, s, sprime] = 1
            elif sprime <= m.vb - initial_band < s:
                fermi[:, :, :, s, sprime] = -1

    return fermi, delta_ea, grad_dea


def precompute_derivs_2d(berry_connections, band_list):
    # shape: beta, alpha1, alpha2, s, sprime, nkx, nky
    nkx, nky = m.nkx, m.nky
    dims = m.dimensions
    derivs = np.zeros((dims, dims, dims, len(band_list), len(band_list), nkx, nky), dtype=np.complex128)
    for beta, alpha1, alpha2 in product(range(dims), repeat=3):
        for i, s in enumerate(band_list):
            for j, sprime in enumerate(band_list):
                if s == sprime:
                    continue
                derivs[beta, alpha1, alpha2, i, j, :, :] = comutederiv(
                    berry_connections, s, sprime, beta, alpha1, alpha2, m.step
                )
    return derivs


def calculate_shg(omega: float, broadning: float, derivs):

    omega_array = np.full(OMEGA_SHAPE, omega + broadning)                        # in Ry
 
    gamma1 = CONST * delta_ea / (2 * omega_array - delta_ea)                     # factor called dE/g in paper times leading constant
    gamma2 = -fermi / np.square(omega_array - delta_ea)                          # factor f/h^2 in paper (-) to account for change in indices in f and h
    gamma3 = -fermi / (omega_array - delta_ea)                                   # factor f/h in paper (index reference is of h, not f, in equation)
#    print('GAMMA1------------------', gamma1)
 #   print('GAMMA3------------------', gamma3)
    if m.dimensions == 1:
        sig = np.full((m.nkx, 1, 1, 1), 0, dtype=np.complex128)
        for s, sprime in product(band_list, repeat=2):                                # runs through index s, s'
            gamma12[:, s, sprime] = gamma1[:, s, sprime] * gamma2[:, s, sprime]
            gamma13[:, s, sprime] = gamma1[:, s, sprime] * gamma3[:, s, sprime]
        # beta, alpha1, alpha2 are spatial coordinates
        for beta, alpha1, alpha2 in product(range(m.dimensions), range(m.dimensions), range(m.dimensions)):
            for s, sprime in product(band_list, band_list):                            # runs through index s, s'
                if s == sprime:
                    continue
                sig[:, beta, alpha1, alpha2] += (
                    (
                        grad_dea[alpha2, :, s, sprime]
                        * comute(berry_connections, sprime, s, beta, alpha1)
                        + grad_dea[alpha1, :, s, sprime]
                        * comute(berry_connections, sprime, s, beta, alpha2)
                    )
                    * gamma12[:, s, sprime]
                    * 0.5
                )

                sig[:, beta, alpha1, alpha2] += (comutederiv(berry_connections, s, sprime, beta, alpha1, alpha2, m.step)) * gamma13[:, s, sprime]

                for r in band_list:                                                   # runs through index r
                    if r in (sprime, s):
                        continue
                    sig[:, beta, alpha1, alpha2] += (
                        -0.25j * gamma1[:, s, sprime] * 
                        (
                            comute3(berry_connections, sprime, s, r, beta, alpha2, alpha1) + 
                            comute3(berry_connections, sprime, s, r, beta, alpha1, alpha2)
                        ) * 
                        gamma3[:, r, sprime] - 
                        (
                            comute3(berry_connections, sprime, s, r, beta, alpha1, alpha2) +
                            comute3(berry_connections, sprime, s, r, beta, alpha2, alpha1)
                        ) * 
                        gamma3[:, s, r]
                    )
    elif m.dimensions == 2:
        nkx, nky = delta_ea.shape[:2]
        sig = np.zeros((nkx, nky, 2, 2, 2), dtype=np.complex128)
    
        gamma12 = gamma1 * gamma2
        gamma13 = gamma1 * gamma3
    
        for beta in range(m.dimensions):
            for alpha1 in range(m.dimensions):
                for alpha2 in range(m.dimensions):
                    for i_s in range(len(band_list)):
                        for j_sprime in range(len(band_list)):
                            s = band_list[i_s]
                            sprime = band_list[j_sprime]
                            if s == sprime:
                                continue
                            # first term
                            sig[:, :, beta, alpha1, alpha2] += 0.5 * (
                                grad_dea[alpha2, :, :, i_s, j_sprime] *
                                comute(berry_connections, sprime, s, beta, alpha1) +
                                grad_dea[alpha1, :, :, i_s, j_sprime] *
                                comute(berry_connections, sprime, s, beta, alpha2)
                            ) * gamma12[:, :, i_s, j_sprime]

                            # precomputed deriv term
                            sig[:, :, beta, alpha1, alpha2] += derivs[beta, alpha1, alpha2, i_s, j_sprime, :, :] * gamma13[:, :, i_s, j_sprime]

                            # sum over r
                            for k_r in range(len(band_list)):
                                r = band_list[k_r]
                                if r in (sprime, s):
                                    continue
                                sig[:, :, beta, alpha1, alpha2] += -0.25j * gamma1[:, :, i_s, j_sprime] * (
                                    comute3(berry_connections, sprime, s, r, beta, alpha2, alpha1) +
                                    comute3(berry_connections, sprime, s, r, beta, alpha1, alpha2)
                                ) * gamma3[:, :, k_r, j_sprime] - (
                                    comute3(berry_connections, sprime, s, r, beta, alpha1, alpha2) +
                                    comute3(berry_connections, sprime, s, r, beta, alpha2, alpha1)
                                ) * gamma3[:, :, i_s, k_r]
#DIMENSIONS ==3 MISSING
    if m.dimensions == 1:
        return (omega, np.sum(sig, axis=0) * VK)
    elif m.dimensions == 2:
        return (omega, np.sum(np.sum(sig, axis=0), axis=0) * VK)
    else:
        return (omega, np.sum(np.sum(np.sum(sig, axis=0), axis=0), axis=0) * VK)



#def run_shg(conduction_band: int, min_band: int = 0, npr: int, energy_max: float = 2.5,
 #           energy_step: float = 0.001, brd: float = 0.01,
  #          logger_name: str = "shg", logger_level: int = logging.INFO, flush: bool = False):

def run_shg(conduction_band: int, npr: int, min_band: int = 0, energy_max: float = 2.5, energy_step: float = 0.001, brd: float = 0.01, logger_name: str = "shg", logger_level: int = logging.INFO, flush: bool = False):
    global gamma1, gamma2, gamma3, gamma12, gamma13, fermi, delta_ea, grad_dea
    global band_list, berry_connections, OMEGA_SHAPE, CONST, VK, initial_band, number_of_bands

    logger = log(logger_name, "SECOND HARMONIC GENERATOR", level=logger_level, flush=flush)
    logger.header()

    if min_band > conduction_band:
        logger.info("Error: Minimum band greater than conduction band!")
        logger.footer()
        exit(1)
    print('npr:', npr)
    initial_band = min_band
    number_of_bands = conduction_band - initial_band + 1
    broadning = brd * 1j

    # -----------------------------
    # 1. Constants
    # -----------------------------
    RY = 13.6056923
    VK = (m.step / (2 * np.pi)) ** m.dimensions

    CONST = np.sqrt(2) * 2 / (2 * np.pi) ** m.dimensions if m.noncolin else 2 * np.sqrt(2) * 2 / (2 * np.pi) ** m.dimensions

    band_list = list(range(conduction_band + 1 - initial_band))
    band_info = list(range(initial_band, conduction_band + 1))

    cb = conduction_band + 1 - initial_band
    if m.dimensions == 1:
        GAMMA_SHAPE = (m.nkx, cb, cb)
        OMEGA_SHAPE = (m.nkx, cb, cb)
        berry_conn_size = m.dimensions * m.nkx * cb**2
        berry_conn_shape = (cb, cb, m.dimensions, m.nkx)
    elif m.dimensions == 2:
        GAMMA_SHAPE = (m.nkx, m.nky, cb, cb)
        OMEGA_SHAPE = (m.nkx, m.nky, cb, cb)
        berry_conn_size = m.dimensions * m.nkx * m.nky * cb**2
        berry_conn_shape = (cb, cb, m.dimensions, m.nkx, m.nky)
    else:
        GAMMA_SHAPE = (m.nkx, m.nky, m.nkz, cb, cb)
        OMEGA_SHAPE = (m.nkx, m.nky, m.nkz, cb, cb)
        berry_conn_size = m.dimensions * m.nkx * m.nky * m.nkz * cb**2
        berry_conn_shape = (cb, cb, m.dimensions, m.nkx, m.nky, m.nkz)

    # -----------------------------
    # 2. Log parameters
    # -----------------------------
    logger.info(f"\tUsing {npr} processes")
    logger.info(f"\n\tList of bands: {band_info}")
    logger.info(f"\tNumber of k-points in each direction: {m.nkx} {m.nky} {m.nkz}")
    logger.info(f"\tNumber of bands: {m.nbnd}")
    logger.info(f"\tk-points step, dk {m.step}")
    logger.info(f"\n\tMaximum energy (Ry): {energy_max}")
    logger.info(f"\tEnergy step (Ry): {energy_step}")
    logger.info(f"\tEnergy broadning (Ry): {np.imag(broadning)}")
    logger.info(f"\tThis is {'noncolinear' if m.noncolin else 'no spin'} calculation")
    logger.info(f"\tConstant in Rydberg units: {np.imag(CONST)}")
    logger.info(f"\tDimensions = {m.dimensions}, Volume/Area in k-space: {VK}\n")

    # -----------------------------
    # 3. Load data and compute gradients
    # -----------------------------
    if m.dimensions == 1:
        grad = Gradient(h=[m.step], acc=2)
    elif m.dimensions == 2:
        grad = Gradient(h=[m.step, m.step], acc=2)
    else:
        grad = Gradient(h=[m.step, m.step, m.step], acc=2)

    bandsfinal = np.load(os.path.join(m.data_dir, "bandsfinal.npy"))
    eigen_array = correct_eigenvalues(bandsfinal)  # <-- keep original single-argument version
    berry_connections = load_berry_connections(conduction_band, berry_conn_size, berry_conn_shape, initial_band)
    fermi, delta_ea, grad_dea = get_fermi_delta_ea_grad_ea(grad, eigen_array, conduction_band - initial_band)

    gamma1 = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma2 = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma3 = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma12 = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma13 = np.zeros(GAMMA_SHAPE, dtype=np.complex128)

    # -----------------------------
    # 4. Precompute derivs
    # -----------------------------
    if m.dimensions == 1:
        derivs = precompute_derivs_1d(berry_connections, band_list)
    elif m.dimensions == 2:
        derivs = precompute_derivs_2d(berry_connections, band_list)
    else:
        derivs = precompute_derivs_3d(delta_ea, grad_dea, berry_connections, band_list)

    # -----------------------------
    # 5. Second Harmonic Generation
    # -----------------------------
    sigma = {}
    omegas = np.arange(0, energy_max + energy_step, energy_step)

    # Multiprocessing: pass derivs
    with Pool(npr) as pool:
        results = pool.starmap(calculate_shg, ((omega, broadning, derivs) for omega in omegas))
        for omega, result in results:
            sigma[omega] = result

    # -----------------------------
    # 6. Check for missing / NaN / zero values
    # -----------------------------
    missing = []
    nan_omegas = []
    zero_omegas = []

    for omega in omegas:
        if omega not in sigma:
            missing.append(omega)
            continue
        val = sigma[omega]
        if not np.isfinite(val).all():
            nan_omegas.append(omega)
        elif np.allclose(val, 0.0):
            zero_omegas.append(omega)

    print("Missing ω:", len(missing))
    print("NaN ω:", len(nan_omegas))
    print("All-zero ω:", len(zero_omegas))

    # -----------------------------
    # 7. Save output (real + imaginary)
    # -----------------------------
    ###########################################################################
    # 5. SAVE OUTPUT
    ###########################################################################
    with open(os.path.join(m.workdir, "sigma2r_vec.dat"), "w") as sigm:
        if m.dimensions == 1:
            sigm.write("# Energy (eV), sigma_xxx\n")
            for omega in np.arange(0, energy_max + energy_step, energy_step):
                outp = "{0:.4f}  {1:.4e}\n"
                sigm.write(
                    outp.format(
                        omega * RY,
                        np.real(sigma[omega][0, 0, 0])
                    )
                )
        elif m.dimensions == 2:
            sigm.write("# Energy (eV), sigma_xxx, sigma_yyy, sigma_xxy, sigma_xyx, sigma_xyy, sigma_yyx, sigma_yxy, sigma_yxx\n")
            for omega in np.arange(0, energy_max + energy_step, energy_step):
                outp = "{0:.4f}  {1:.4e}  {2:.4e}  {3:.4e}  {4:.4e}  {5:.4e}  {6:.4e}  {7:.4e}  {8:.4e}\n"
                sigm.write(
                    outp.format(
                        omega * RY,
                        np.real(sigma[omega][0, 0, 0]),
                        np.real(sigma[omega][1, 1, 1]),
                        np.real(sigma[omega][0, 0, 1]),
                        np.real(sigma[omega][0, 1, 0]),
                        np.real(sigma[omega][0, 1, 1]),
                        np.real(sigma[omega][1, 1, 0]),
                        np.real(sigma[omega][1, 0, 1]),
                        np.real(sigma[omega][1, 0, 0]),
                    )
                )
        else:
            sigm.write("# Energy (eV), sigma_xxx, sigma_yyy, sigma_zzz, sigma_xxy, sigma_xxz, sigma_xyx, sigma_xzx, sigma_yxx, \
                      sigma_zxx, sigma_xyy, sigma_zyy, sigma_yyx, sigma_yyz, sigma_yxy, sigma_yzy, \
                      sigma_xzz, sigma_yzz, sigma_zzx, sigma_zzy, sigma_zxz, sigma_zyz, sigma_xyz, \
                      sigma_zxy, sigma_yzx, sigma_xzy, sigma_zyx, sigma_yxz    \n")
            for omega in np.arange(0, energy_max + energy_step, energy_step):
                outp = "{0:.4f}  {1:.4e}  {2:.4e}  {3:.4e}  {4:.4e}  {5:.4e}  {6:.4e}  {7:.4e}  {8:.4e} \
                        {9:.4e}  {10:.4e}  {11:.4e}  {12:.4e}  {13:.4e}  {14:.4e}  {15:.4e}  {16:.4e}  {17:.4e}  {18:.4e} \
                        {19:.4e}  {20:.4e}  {21:.4e}  {22:.4e}  {23:.4e}  {24:.4e}  {25:.4e}  {26:.4e}  {27:.4e}\n"
                sigm.write(
                    outp.format(
                        omega * RY,
                        np.real(sigma[omega][0, 0, 0]),
                        np.real(sigma[omega][1, 1, 1]),
                        np.real(sigma[omega][2, 2, 2]),
                        np.real(sigma[omega][0, 0, 1]),
                        np.real(sigma[omega][0, 0, 2]),
                        np.real(sigma[omega][0, 1, 0]),
                        np.real(sigma[omega][0, 2, 0]),
                        np.real(sigma[omega][1, 0, 0]),
                        np.real(sigma[omega][2, 0, 0]),
                        np.real(sigma[omega][0, 1, 1]),
                        np.real(sigma[omega][2, 1, 1]),
                        np.real(sigma[omega][1, 1, 0]),
                        np.real(sigma[omega][1, 1, 2]),
                        np.real(sigma[omega][1, 0, 1]),
                        np.real(sigma[omega][1, 2, 1]),
                        np.real(sigma[omega][0, 2, 2]),
                        np.real(sigma[omega][1, 2, 2]),
                        np.real(sigma[omega][2, 2, 0]),
                        np.real(sigma[omega][2, 2, 1]),
                        np.real(sigma[omega][2, 0, 2]),
                        np.real(sigma[omega][2, 1, 2]),
                        np.real(sigma[omega][0, 1, 2]),
                        np.real(sigma[omega][2, 0, 1]),
                        np.real(sigma[omega][1, 2, 0]),
                        np.real(sigma[omega][0, 2, 1]),
                        np.real(sigma[omega][2, 1, 0]),
                        np.real(sigma[omega][1, 0, 2]),
                    )
                )

    logger.info("\tReal part of SHG saved to file sigma2r.dat")

    with open(os.path.join(m.workdir, "sigma2i_vec.dat"), "w") as sigm:
        if m.dimensions == 1:
            sigm.write("# Energy (eV), sigma_xxx\n")
            for omega in np.arange(0, energy_max + energy_step, energy_step):
                outp = "{0:.4f}  {1:.4e}\n"
                sigm.write(
                    outp.format(
                        omega * RY,
                        np.imag(sigma[omega][0, 0, 0]),
                    )
                )
        elif m.dimensions == 2:
            sigm.write("# Energy (eV), sigma_xxx, sigma_yyy, sigma_xxy, sigma_xyx, sigma_xyy, sigma_yyx, sigma_yxy, sigma_yxx\n")
            for omega in np.arange(0, energy_max + energy_step, energy_step):
                outp = "{0:.4f}  {1:.4e}  {2:.4e}  {3:.4e}  {4:.4e}  {5:.4e}  {6:.4e}  {7:.4e}  {8:.4e}\n"
                sigm.write(
                    outp.format(
                        omega * RY,
                        np.imag(sigma[omega][0, 0, 0]),
                        np.imag(sigma[omega][1, 1, 1]),
                        np.imag(sigma[omega][0, 0, 1]),
                        np.imag(sigma[omega][0, 1, 0]),
                        np.imag(sigma[omega][0, 1, 1]),
                        np.imag(sigma[omega][1, 1, 0]),
                        np.imag(sigma[omega][1, 0, 1]),
                        np.imag(sigma[omega][1, 0, 0]),
                    )
                )
        else:
            sigm.write("# Energy (eV), sigma_xxx, sigma_yyy, sigma_zzz, sigma_xxy, sigma_xxz, sigma_xyx, sigma_xzx, sigma_yxx, \
                      sigma_zxx, sigma_xyy, sigma_zyy, sigma_yyx, sigma_yyz, sigma_yxy, sigma_yzy, \
                      sigma_xzz, sigma_yzz, sigma_zzx, sigma_zzy, sigma_zxz, sigma_zyz, sigma_xyz, \
                      sigma_zxy, sigma_yzx, sigma_xzy, sigma_zyx, sigma_yxz    \n")
            for omega in np.arange(0, energy_max + energy_step, energy_step):
                outp = "{0:.4f}  {1:.4e}  {2:.4e}  {3:.4e}  {4:.4e}  {5:.4e}  {6:.4e}  {7:.4e}  {8:.4e} \
                        {9:.4e}  {10:.4e}  {11:.4e}  {12:.4e}  {13:.4e}  {14:.4e}  {15:.4e}  {16:.4e}  {17:.4e}  {18:.4e} \
                        {19:.4e}  {20:.4e}  {21:.4e}  {22:.4e}  {23:.4e}  {24:.4e}  {25:.4e}  {26:.4e}  {27:.4e}\n"
                sigm.write(
                    outp.format(
                        omega * RY,
                        np.real(sigma[omega][0, 0, 0]),
                        np.real(sigma[omega][1, 1, 1]),
                        np.real(sigma[omega][2, 2, 2]),
                        np.real(sigma[omega][0, 0, 1]),
                        np.real(sigma[omega][0, 0, 2]),
                        np.real(sigma[omega][0, 1, 0]),
                        np.real(sigma[omega][0, 2, 0]),
                        np.real(sigma[omega][1, 0, 0]),
                        np.real(sigma[omega][2, 0, 0]),
                        np.real(sigma[omega][0, 1, 1]),
                        np.real(sigma[omega][2, 1, 1]),
                        np.real(sigma[omega][1, 1, 0]),
                        np.real(sigma[omega][1, 1, 2]),
                        np.real(sigma[omega][1, 0, 1]),
                        np.real(sigma[omega][1, 2, 1]),
                        np.real(sigma[omega][0, 2, 2]),
                        np.real(sigma[omega][1, 2, 2]),
                        np.real(sigma[omega][2, 2, 0]),
                        np.real(sigma[omega][2, 2, 1]),
                        np.real(sigma[omega][2, 0, 2]),
                        np.real(sigma[omega][2, 1, 2]),
                        np.real(sigma[omega][0, 1, 2]),
                        np.real(sigma[omega][2, 0, 1]),
                        np.real(sigma[omega][1, 2, 0]),
                        np.real(sigma[omega][0, 2, 1]),
                        np.real(sigma[omega][2, 1, 0]),
                        np.real(sigma[omega][1, 0, 2]),
                    )
                )
    logger.info("\tImaginary part of SHG saved to file sigma2i.dat")

    ###################################################################################
    # Finished
    ###################################################################################

    logger.footer()


if __name__ == "__main__":
    run_shg(9, log("shg", "SECOND HARMONIC GENERATION", "version", logging.DEBUG), npr=40)
