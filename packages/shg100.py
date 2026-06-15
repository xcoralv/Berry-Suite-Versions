from multiprocessing import Pool, Array, Manager
from itertools import product
from typing import Tuple
#from joblib import Parallel, delayed

import os
import ctypes
import logging

from findiff import Gradient

import numpy as np

from berry import log
from berry._subroutines.comutator import comute_vec, comutederiv_vec, comute3_vec

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

def calculate_shg(omega: float, broadning: float):

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
        sig = np.full((m.nkx, m.nky, 2, 2, 2), 0, dtype=np.complex128)               # matrix sig_xxx,sig_xxy,...,sig_yyx,sig_yyy
        for s, sprime in product(band_list, repeat=2):                                # runs through index s, s'
            gamma12[:, :, s, sprime] = gamma1[:, :, s, sprime] * gamma2[:, :, s, sprime]
            gamma13[:, :, s, sprime] = gamma1[:, :, s, sprime] * gamma3[:, :, s, sprime]
        # beta, alpha1, alpha2 are spatial coordinates
        #components = [(beta, alpha1, alpha2) for beta in range(m.dimensions)
         #                                 for alpha1 in range(m.dimensions)
          #                                for alpha2 in range(m.dimensions)]
        args = [(beta, alpha1, alpha2)
            for beta in range(m.dimensions)
            for alpha1 in range(m.dimensions)
            for alpha2 in range(m.dimensions)]

        for (beta, alpha1, alpha2) in args:
            partial_result = vectorize(beta, alpha1, alpha2)
            sig[:, :, beta, alpha1, alpha2] = partial_result

#DIMENSIONS ==3 MISSING
    if m.dimensions == 1:
        return (omega, np.sum(sig, axis=0) * VK)
    elif m.dimensions == 2:
        return (omega, np.sum(np.sum(sig, axis=0), axis=0) * VK)
    else:
        return (omega, np.sum(np.sum(np.sum(sig, axis=0), axis=0), axis=0) * VK)

'''
def vectorize(beta, alpha1, alpha2):
    """
    Vectorized calculation of SHG contributions for a given tensor component.
    All 'pair' and 'triple' axes are moved to the last dimension to allow broadcasting.
    """
    # 1. Initialize output
    sig = np.zeros((m.nkx, m.nky), dtype=np.complex128)

    # 2. Generate all s != s' pairs
    pairs = np.array([(s, sprime) for s in band_list for sprime in band_list if s != sprime])
    s_indices = pairs[:, 0]
    sprime_indices = pairs[:, 1]

    # 3. Stack gamma factors and move 'pair' axis to last
    gamma12_stack = gamma12[:, :, s_indices, sprime_indices]    # (nkx, nky, Npairs)
    gamma12_stack = np.moveaxis(gamma12_stack, 2, -1)           # ensure Npairs last
    gamma13_stack = gamma13[:, :, s_indices, sprime_indices]
    gamma13_stack = np.moveaxis(gamma13_stack, 2, -1)

    # 4. Stack grad_dea and move 'pair' axis to last
    g1 = grad_dea[alpha2, :, :, s_indices, sprime_indices]       # (Npairs, nkx, nky)
    g1 = np.moveaxis(g1, 0, -1)                                 # (nkx, nky, Npairs)
    g2 = grad_dea[alpha1, :, :, s_indices, sprime_indices]
    g2 = np.moveaxis(g2, 0, -1)

    # 5. Vectorized commutators
    comute_1 = comute_vec(berry_connections, sprime_indices, s_indices, beta, alpha1)  # (nkx, nky, Npairs)
    comute_2 = comute_vec(berry_connections, sprime_indices, s_indices, beta, alpha2)
    comutederiv_term = comutederiv_vec(berry_connections, s_indices, sprime_indices, beta, alpha1, alpha2, m.step)

    # 6. First contributions (sum over Npairs)
    term1 = 0.5 * (g1 * comute_1 + g2 * comute_2) * gamma12_stack
    term2 = comutederiv_term * gamma13_stack
    sig += np.sum(term1 + term2, axis=-1)

    # 7. Now handle all r != s,s' for comute3
    triples = []
    for s, sprime in pairs:
        r_list = [r for r in band_list if r not in (s, sprime)]
        s_arr = np.full(len(r_list), s)
        sprime_arr = np.full(len(r_list), sprime)
        r_arr = np.array(r_list)
        triples.append(np.stack([s_arr, sprime_arr, r_arr], axis=1))

    triples = np.concatenate(triples, axis=0)
    s_arr, sprime_arr, r_arr = triples[:, 0], triples[:, 1], triples[:, 2]

    # 8. Vectorized comute3 terms
    comute3_term1 = comute3_vec(berry_connections, sprime_arr, s_arr, r_arr, beta, alpha2, alpha1)  # (nkx, nky, Ntriples)
    comute3_term2 = comute3_vec(berry_connections, sprime_arr, s_arr, r_arr, beta, alpha1, alpha2)

    # 9. Align gamma factors for triples
    gamma1_factors = gamma1[:, :, s_arr, sprime_arr]    # (nkx, nky, Ntriples)
    gamma1_factors = np.moveaxis(gamma1_factors, 2, -1)  # move Ntriples last

    gamma3_r_sprime = gamma3[:, :, r_arr, sprime_arr]
    gamma3_r_sprime = np.moveaxis(gamma3_r_sprime, 2, -1)

    gamma3_s_r = gamma3[:, :, s_arr, r_arr]
    gamma3_s_r = np.moveaxis(gamma3_s_r, 2, -1)

    # 10. Compute comute3 contributions and sum over Ntriples
    sig += np.sum(
        -0.25j * gamma1_factors * comute3_term1 * gamma3_r_sprime
        - comute3_term2 * gamma3_s_r,
        axis=-1
    )

    print("g1 shape:", g1.shape)
    print("comute_1 shape:", comute_1.shape)
    print("gamma12_stack shape:", gamma12_stack.shape)

    return sig
'''

def vectorize(beta, alpha1, alpha2):
    """
    Fully vectorized computation of sigma with aligned gamma factors and commutators.
    """
    sig = np.zeros((m.nkx, m.nky), dtype=np.complex128)

    # 1. Generate all s != s' pairs
    pairs = np.array([(s, sprime) for s in band_list for sprime in band_list if s != sprime])
    s_indices = pairs[:, 0]
    sprime_indices = pairs[:, 1]

    # 2. Stack gamma factors for s != s' pairs
    gamma12_stack = gamma12[:, :, s_indices, sprime_indices]  # (nkx, nky, Npairs)
    gamma13_stack = gamma13[:, :, s_indices, sprime_indices]

    # 3. Stack grad_dea
    g1 = grad_dea[alpha2, :, :, s_indices, sprime_indices]  # (nkx, nky, Npairs)
    g2 = grad_dea[alpha1, :, :, s_indices, sprime_indices]

    # Align g1 and g2 to match comute and gamma
    g1 = np.moveaxis(g1, 0, -1)  # from (90, 10, 10) → (10, 10, 90)
    g2 = np.moveaxis(g2, 0, -1)

    # 4. Vectorized commutators / derivative terms
    comute_1 = comute_vec(berry_connections, sprime_indices, s_indices, beta, alpha1)  # (nkx, nky, Npairs)
    comute_2 = comute_vec(berry_connections, sprime_indices, s_indices, beta, alpha2)
    comutederiv_term = comutederiv_vec(berry_connections, s_indices, sprime_indices, beta, alpha1, alpha2, m.step)

    # 5. First two contributions
    term1 = 0.5 * (g1 * comute_1 + g2 * comute_2) * gamma12_stack
    term2 = comutederiv_term * gamma13_stack
    sig += np.sum(term1 + term2, axis=-1)  # sum over Npairs

    # 6. Handle all r != s,s' for comute3 with proper gamma alignment
    triples_list = []
    gamma1_list = []
    gamma3_r_sprime_list = []
    gamma3_s_r_list = []

    for s, sprime in pairs:
        r_list = [r for r in band_list if r not in (s, sprime)]
        n_r = len(r_list)

        # Build arrays for this block
        s_arr_block = np.full(n_r, s)
        sprime_arr_block = np.full(n_r, sprime)
        r_arr_block = np.array(r_list)

        triples_list.append(np.stack([s_arr_block, sprime_arr_block, r_arr_block], axis=1))

        # Stack gamma factors for this block
        gamma1_list.append(gamma1[:, :, s_arr_block, sprime_arr_block])       # (nkx, nky, n_r)
        gamma3_r_sprime_list.append(gamma3[:, :, r_arr_block, sprime_arr_block])
        gamma3_s_r_list.append(gamma3[:, :, s_arr_block, r_arr_block])

    # Concatenate all blocks
    triples = np.concatenate(triples_list, axis=0)
    s_arr, sprime_arr, r_arr = triples[:, 0], triples[:, 1], triples[:, 2]

    gamma1_factors = np.concatenate(gamma1_list, axis=-1)         # (nkx, nky, Ntriples)
    gamma3_r_sprime = np.concatenate(gamma3_r_sprime_list, axis=-1)
    gamma3_s_r = np.concatenate(gamma3_s_r_list, axis=-1)

    # 7. Vectorized comute3 terms
    comute3_term1 = comute3_vec(berry_connections, sprime_arr, s_arr, r_arr, beta, alpha2, alpha1)
    comute3_term2 = comute3_vec(berry_connections, sprime_arr, s_arr, r_arr, beta, alpha1, alpha2)

    # 8. Compute comute3 contribution with correct alignment
    sig += np.sum(
        -0.25j * gamma1_factors * comute3_term1 * gamma3_r_sprime
        - comute3_term2 * gamma3_s_r,
        axis=-1
    )

    return sig



def run_shg(conduction_band: int, min_band: int = 0, npr: int = 1, energy_max: float = 2.5, energy_step: float = 0.001, brd: float = 0.01, logger_name: str = "shg", logger_level: int = logging.INFO, flush: bool = False):
    global gamma1, gamma2, gamma3, gamma12, gamma13, fermi, delta_ea, grad_dea, band_list, berry_connections, OMEGA_SHAPE, CONST, VK, initial_band, number_of_bands
    logger = log(logger_name, "SECOND HARMONIC GENERATOR", level=logger_level, flush=flush)

    logger.header()

    if min_band > conduction_band:
        logger.info("Error: Minimum band greater than conduction band!")
        logger.footer()
        exit(1)

    initial_band = min_band
    number_of_bands = conduction_band - initial_band + 1
    broadning = brd*1j

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ###########################################################################
    RY    = 13.6056923                                                          # Conversion factor from Ry to eV
    VK    = (m.step / (2 * np.pi)) ** m.dimensions                                # element of volume in k-space in units of bohr^-1
                                                                                
    if m.noncolin:
        CONST = np.sqrt(2) * 2 / (2 * np.pi) ** m.dimensions
    else:
        CONST = 2 * np.sqrt(2) * 2 / (2 * np.pi) ** m.dimensions                # = -2e^3/hslash 1/(2pi)^d     in Rydberg units
                                                                                # the 2e comes from having two electrons per band
                                                                                # another minus comes from the negative charge

    band_list   = list(range(conduction_band + 1 - initial_band))
    band_info   = list(range(initial_band, conduction_band + 1))

    #TODO: Add docstring with these comments
    # Maximum energy (Ry)
    # Energy step (Ry)
    # energy broading (Ry)

    cb = conduction_band + 1 - initial_band
    if m.dimensions == 1:
        GAMMA_SHAPE = (m.nkx, cb, cb)
        OMEGA_SHAPE = (m.nkx, cb, cb)
        berry_conn_size  = m.dimensions * m.nkx * (cb) ** 2
        berry_conn_shape = (cb, cb, m.dimensions, m.nkx)
    elif m.dimensions == 2:
        GAMMA_SHAPE = (m.nkx, m.nky, cb, cb)
        OMEGA_SHAPE = (m.nkx, m.nky, cb, cb)
        berry_conn_size  = m.dimensions * m.nkx * m.nky * (cb) ** 2
        berry_conn_shape = (cb, cb, m.dimensions, m.nkx, m.nky)
    else:
        GAMMA_SHAPE = (m.nkx, m.nky, m.nkz, cb, cb)
        OMEGA_SHAPE = (m.nkx, m.nky, m.nkz, cb, cb)
        berry_conn_size  = m.dimensions * m.nkx * m.nky * m.nkz * (cb) ** 2
        berry_conn_shape = (cb, cb, m.dimensions, m.nkx, m.nky, m.nkz)


    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ###########################################################################
    logger.info(f"\tUsing {npr} processes")
    
    logger.info(f"\n\tList of bands: {band_info}")
    logger.info(f"\tNumber of k-points in each direction: {m.nkx} {m.nky} {m.nkz}")
    logger.info(f"\tNumber of bands: {m.nbnd}")
    logger.info(f"\tk-points step, dk {m.step}")                                      # Defines the step for gradient calculation dk

    logger.info(f"\n\tMaximum energy (Ry): {energy_max}")
    logger.info(f"\tEnergy step (Ry): {energy_step}")
    logger.info(f"\tEnergy broadning (Ry): {np.imag(broadning)}")
    if m.noncolin:
        logger.info(f"\tThis is a noncolinear calculation.")
        logger.info(f"\tConstant -2e^3/hslash 1/(2pi)^d in Rydberg units: {np.imag(CONST)}")
    else:
        logger.info(f"\tThis is a no spin calculation.")
        logger.info(f"\tConstant -4e^3/hslash 1/(2pi)^d in Rydberg units: {np.imag(CONST)}")
    logger.info(f"\tNumber of dimensions d = {m.dimensions}")
    logger.info(f"\tVolume (area) in k space: {VK}\n")

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    if m.dimensions == 1:
        grad = Gradient(h=[m.step], acc=2)           # Defines gradient function in 1D
    elif m.dimensions == 2:
        grad = Gradient(h=[m.step, m.step], acc=2)           # Defines gradient function in 2D
    else:
        grad = Gradient(h=[m.step, m.step, m.step], acc=2)           # Defines gradient function in 3D

    bandsfinal                  = np.load(os.path.join(m.data_dir, "bandsfinal.npy"))
    eigen_array                 = correct_eigenvalues(bandsfinal)
    berry_connections           = load_berry_connections(conduction_band, berry_conn_size, berry_conn_shape, initial_band)
    fermi, delta_ea, grad_dea   = get_fermi_delta_ea_grad_ea(grad, eigen_array, conduction_band - initial_band)

    gamma1                      = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma2                      = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma3                      = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma12                     = np.zeros(GAMMA_SHAPE, dtype=np.complex128)
    gamma13                     = np.zeros(GAMMA_SHAPE, dtype=np.complex128)

    ###########################################################################
    # 4. SECONG HARMONIC GENERATION
    ###########################################################################
    sigma = {}
    with Pool(npr) as pool:
        results = pool.starmap(calculate_shg, ((omega, broadning) for omega in np.arange(0, energy_max + energy_step, energy_step)))
        for omega, result in results:
            sigma[omega] = result

#    ALTERATION HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    omegas = np.arange(0, energy_max + energy_step, energy_step)

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
#    ALTERATION HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


#    sigma = Manager().dict()
 #   with Pool(npr) as pool:
  #      results = pool.starmap(calculate_shg_and_store, ((omega, broadning) for omega in np.arange(0, energy_max + energy_step, energy_step)))
   # sigma = dict(results)



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
