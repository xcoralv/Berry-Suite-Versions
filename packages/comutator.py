""" Module to include some commutators for condutivity calculations """
import time
from findiff import Gradient
import numpy as np


comute3_total_time = 0.0
comutederiv_time = 0.0
comute_time = 0.0


def comute(berryConnection, sprime, s, beta, alpha):
    """ Commute two Berry connections."""
    global comute_time
    start = time.perf_counter()
    e = (
        berryConnection[sprime][s][beta] * berryConnection[s][sprime][alpha]
        - berryConnection[sprime][s][alpha] * berryConnection[s][sprime][beta]
    )
    end = time.perf_counter()
    comute_time += (end - start)
    return e



def comute_vec(berry_conn, sprime_arr, s_arr, beta, alpha):
    """
    Fully vectorized commutator: [A,B] = A*B - B*A
    berry_conn shape: (cb, cb, dim, nkx[, nky, nkz])
    sprime_arr, s_arr: arrays of length Npairs
    Returns: array of shape (nkx[, nky, nkz], Npairs)
    """
    # Extract arrays for all pairs at once
    A = berry_conn[sprime_arr, s_arr, beta]    # shape (Npairs, nkx, nky)
    B = berry_conn[s_arr, sprime_arr, alpha]   # shape (Npairs, nkx, nky)
    C = berry_conn[sprime_arr, s_arr, alpha]
    D = berry_conn[s_arr, sprime_arr, beta]

    # Compute commutator
    result = (A * B - C * D)  # shape (Npairs, nkx, nky)
    
    # Move Npairs to last axis for consistency with vectorize()
    result = np.moveaxis(result, 0, -1)  # shape (nkx, nky, Npairs)

    return result





def comute3_vec(berry_conn, sprime_arr, s_arr, r_arr, beta, alpha2, alpha1):
    """
    Vectorized commutator of three Berry connections
    berry_conn shape: (cb, cb, dim, nkx[, nky, nkz])
    sprime_arr, s_arr, r_arr: arrays of shape (Ntriples,)
    Returns: array of shape (nkx[, nky, nkz], Ntriples)
    """
    # Pull arrays for all triples
    A = berry_conn[sprime_arr, s_arr, beta]   # (Ntriples, nkx, nky)
    B = berry_conn[s_arr, r_arr, alpha2]      # (Ntriples, nkx, nky)
    C = berry_conn[r_arr, sprime_arr, alpha1]# (Ntriples, nkx, nky)

    D = berry_conn[sprime_arr, r_arr, alpha1]
    E = berry_conn[r_arr, s_arr, alpha2]
    F = berry_conn[s_arr, sprime_arr, beta]

    result = (A * B * C + D * E * F)  # shape (Ntriples, nkx, nky)
    result = np.moveaxis(result, 0, -1)  # (nkx, nky, Ntriples)

    return result


'''
def comute3_vec(berry_conn, sprime_arr, s_arr, r_arr, beta, alpha2, alpha1, gamma1_arr, gamma3_r_sprime, gamma3_s_r):
    """
    Vectorized version of the triple commutator contribution for SHG.
    Matches the loop:
        sig += -0.25j * gamma1[s,s'] * (comute3(sp,s,r,b,a2,a1) + comute3(sp,s,r,b,a1,a2)) * gamma3[r,s']
             - (comute3(sp,s,r,b,a1,a2) + comute3(sp,s,r,b,a2,a1)) * gamma3[s,r]

    Parameters
    ----------
    berry_conn : np.ndarray
        Berry connection array, shape (cb, cb, dim, nkx[, nky, nkz])
    sprime_arr, s_arr, r_arr : np.ndarray
        Arrays of length Ntriples with indices of s', s, r
    beta, alpha2, alpha1 : int
        Spatial indices for tensor component
    gamma1_arr : np.ndarray
        Pre-aligned gamma1 factors, shape (nkx, nky[, nkz], Ntriples)
    gamma3_r_sprime : np.ndarray
        Pre-aligned gamma3[r, s'], shape (nkx, nky[, nkz], Ntriples)
    gamma3_s_r : np.ndarray
        Pre-aligned gamma3[s, r], shape (nkx, nky[, nkz], Ntriples)

    Returns
    -------
    np.ndarray
        Contribution to sig, shape (nkx, nky[, nkz])
    """
    # Pull Berry connections for all triples
    A = berry_conn[sprime_arr, s_arr, beta]      # (Ntriples, nkx, nky)
    B = berry_conn[s_arr, r_arr, alpha2]
    C = berry_conn[r_arr, sprime_arr, alpha1]

    D = berry_conn[sprime_arr, s_arr, beta]
    E = berry_conn[s_arr, r_arr, alpha1]
    F = berry_conn[r_arr, sprime_arr, alpha2]

    # Move Ntriples axis last for broadcasting
    A = np.moveaxis(A, 0, -1)
    B = np.moveaxis(B, 0, -1)
    C = np.moveaxis(C, 0, -1)
    D = np.moveaxis(D, 0, -1)
    E = np.moveaxis(E, 0, -1)
    F = np.moveaxis(F, 0, -1)

    # Compute the two commutator terms exactly like the loop
    term1 = A * B * C + D * E * F                  # (nkx, nky, Ntriples)
    term2 = B * C * A + E * F * D                  # same shape; corresponds to swapped indices in loop

    # Apply gamma factors with correct broadcasting
    contrib = (-0.25j * gamma1_arr * term1 * gamma3_r_sprime) - (term2 * gamma3_s_r)

    # Sum over Ntriples
    sig_contrib = np.sum(contrib, axis=-1)        # (nkx, nky)

    return sig_contrib
'''




def comute3(berryConnection, sprime, s, r, beta, alpha2, alpha1):
    """ Commute three Berry connections."""
    global comute3_total_time
    start = time.perf_counter()
    e = (
        berryConnection[sprime][s][beta]
        * berryConnection[s][r][alpha2]
        * berryConnection[r][sprime][alpha1]
        + berryConnection[sprime][r][alpha1]
        * berryConnection[r][s][alpha2]
        * berryConnection[s][sprime][beta]
    )
    end = time.perf_counter()
    comute3_total_time += (end-start)

    return e


def deriv(berryConnection, s, sprime, alpha1, alpha2, dk):
    """ Derivative of the Berry connection."""
    grad = Gradient(h=[dk, dk], acc=2)  # Defines gradient function in 2D

    a = grad(berryConnection[s][sprime][alpha1])

    e = (
        a[alpha2]
        - 1j
        * (berryConnection[s][s][alpha2] - berryConnection[sprime][sprime][alpha2])
        * berryConnection[s][sprime][alpha1]
    )

    return e


def deriv_updated(berryConnection, s, sprime, alpha1, alpha2, grad):
    """Derivative of the Berry connection (Gradient precomputed)."""
    a = grad(berryConnection[s][sprime][alpha1])
    e = a[alpha2] - 1j * (berryConnection[s][s][alpha2] - berryConnection[sprime][sprime][alpha2]) * berryConnection[s][sprime][alpha1]
    return e



def comutederiv_vec(berry_conn, s_arr, sprime_arr, beta, alpha1, alpha2, step):
    """
    Fully vectorized commutator with derivative
    berry_conn shape: (cb, cb, dim, nkx[, nky, nkz])
    Returns shape: (nkx[, nky, nkz], Npairs)
    """
    # Number of pairs and k-point shape
    Npairs = len(s_arr)
    shape = berry_conn.shape[3:]  # e.g., (nkx, nky)
    nk_dims = len(shape)

    # Initialize result
    result = np.zeros(shape + (Npairs,), dtype=np.complex128)

    # Create Gradient object for all k-points
    grad = Gradient(h=[step]*nk_dims, acc=2)

    # Loop over spatial components only (usually small, 1-3)
    for i in range(Npairs):
        s = s_arr[i]
        sprime = sprime_arr[i]

        # Compute derivative term for all k-points at once
        a = grad(berry_conn[s, sprime, alpha1])  # returns array of shape = nk dims
        deriv_term = a[alpha2] - 1j*(berry_conn[s, s, alpha2] - berry_conn[sprime, sprime, alpha2])*berry_conn[s, sprime, alpha1]

        # Commute: [B, dB] = B_s's * deriv - deriv * B_ss'
        result[..., i] = berry_conn[sprime, s, beta] * deriv_term - deriv_term * berry_conn[s, sprime, beta]

    return result



#To switch to Gradient precomputation, assign second comutederiv as "comutederiv"

#first comutederiv
def comutederiv_old(berryConnection, s, sprime, beta, alpha1, alpha2, dk):
    """ Commute Berry connection and a derivative."""
    global comutederiv_time
    start = time.perf_counter()
    e = (
        berryConnection[sprime][s][beta]
        * deriv(berryConnection, s, sprime, alpha1, alpha2, dk)
        - deriv(berryConnection, sprime, s, alpha1, alpha2, dk)
        * berryConnection[s][sprime][beta]
    )
    end = time.perf_counter()
    comutederiv_time += end - start
    return e


#second comutederiv
def comutederiv(berryConnection, s, sprime, beta, alpha1, alpha2, dk):
    """ Commute Berry connection and a derivative."""
    global comutederiv_time
    start = time.perf_counter()

    # Initialize once outside
    grad = Gradient(h=[dk, dk], acc=2)

    e = (
        berryConnection[sprime][s][beta]
        * deriv_updated(berryConnection, s, sprime, alpha1, alpha2, grad)
        - deriv_updated(berryConnection, sprime, s, alpha1, alpha2, grad)
        * berryConnection[s][sprime][beta]
    )
    end = time.perf_counter()
    comutederiv_time += end - start
    return e

from numba import njit, prange

@njit
def finite_diff_2d(arr, axis, dk):
    nkx, nky = arr.shape
    grad = np.zeros((nkx, nky), dtype=np.complex128)
    
    if axis == 0:  # derivative along x
        for i in range(1, nkx-1):
            for j in range(nky):
                grad[i,j] = (arr[i+1,j] - arr[i-1,j]) / (2*dk)
        # edges
        grad[0,:] = (arr[1,:] - arr[0,:]) / dk
        grad[-1,:] = (arr[-1,:] - arr[-2,:]) / dk
    else:  # axis == 1, derivative along y
        for i in range(nkx):
            for j in range(1, nky-1):
                grad[i,j] = (arr[i,j+1] - arr[i,j-1]) / (2*dk)
        grad[:,0] = (arr[:,1] - arr[:,0]) / dk
        grad[:,-1] = (arr[:,-1] - arr[:,-2]) / dk
    return grad




@njit(parallel=True)
def comutederiv_numba(berryConnection, s, sprime, beta, alpha1, alpha2, dk):
    nkx, nky = berryConnection.shape[3], berryConnection.shape[4]
    e = np.zeros((nkx, nky), dtype=np.complex128)

    # finite difference along axis alpha2
    grad_arr = finite_diff_2d(berryConnection[s, sprime, alpha1], alpha2, dk)

    for i in prange(nkx):
        for j in range(nky):
            e[i,j] = berryConnection[sprime, s, beta, i, j] * grad_arr[i,j] \
                     - grad_arr[i,j] * berryConnection[s, sprime, beta, i, j]
    return e
