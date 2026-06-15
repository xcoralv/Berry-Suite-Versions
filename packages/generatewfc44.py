from typing import Optional
import time
import os
import logging
import subprocess
import sys
import numpy as np
import cProfile
from berry import log
from multiprocessing import Pool
from collections import Counter
import math
from tempfile import TemporaryDirectory

try:
    import berry._subroutines.loadmeta as m
    import berry._subroutines.loaddata as d
except:
    pass


class WfcGenerator:
    def __init__(self, 
                 nk_points: Optional[int] = None , 
                 bands: Optional[int] = None, 
                 npr: Optional[int] = None , 
                 logger_name: str = "genwfc", 
                 logger_level: int = logging.INFO, 
                 flush: bool = False
                ):
        
        if bands is not None and nk_points is None:
            raise ValueError("To generate a wavefunction for a single band, you must specify the k-point.")
        self.npr = npr if npr is not None else m.npr
        os.system("mkdir -p " + m.wfcdirectory)

        if nk_points is None:
            self.nk_points = range(m.nks)
        elif  bands is None:
            self.nk_points = nk_points
            self.bands = range(m.nbnd)
        else:
            self.nk_points = nk_points
            self.bands = bands
        self.ref_name = m.refname
        self.logger = log(logger_name, "GENERATE WAVE FUNCTIONS", level=logger_level, flush=flush)


    def run(self):
        # prints header on the log file
        self.logger.header()

        # Logs the parameters for the run
        self._log_run_params()

        # Sets the program used for converting wavefunctions to the real space
        if m.noncolin:
            self.k2r_program = "wfck2rFR.x"
            self.logger.info("\tNoncolinear calculation, will use wfck2rFR.x")
        else:
            self.k2r_program = "wfck2r.x"
            self.logger.info("\tNonrelativistic calculation, will use wfck2r.x")

        # Set which k-points and bands will use (for debuging)
        if isinstance(self.nk_points, range):
        
            self.logger.info("\n\tWill run for all k-points and bands")
            self.logger.info(f"\tThere are {m.nks} k-points and {m.nbnd} bands.\n")
            print("Task list:", [(nk, 0, m.nbnd) for nk in self.nk_points])
            with Pool(processes = self.npr) as pool:
                pool.starmap(self._wfck2r, [(nk, 0, m.nbnd) for nk in self.nk_points])

        
              
        else:
            if isinstance(self.bands, range):
                self.logger.info(f"\tWill run for k-point {self.nk_points} and all bands")
                self.logger.info(f"\tThere are {m.nks} k-points and {m.nbnd} bands.\n")

                self.logger.info(f"\tCalculating wfc for k-point {self.nk_points}")
                self._wfck2r(self.nk_points, 0, m.nbnd)
            else:
                self.logger.info(f"\tWill run just for k-point {self.nk_points} and band {self.bands}.\n")
                self._wfck2r(self.nk_points, self.bands, 1)

        self.logger.info("\n\tRemoving temporary file 'tmp'")
        self.logger.info(f"\tRemoving quantum expresso output file '{m.wfck2r}'")

        self.logger.footer()

    def byte_entropy(self, byte_data: bytes) -> float:
        """
        Compute Shannon entropy (bits per byte) of raw byte data.
        0 = no randomness, 8 = fully random/incompressible.
        """
        if len(byte_data) == 0:
            return 0.0
        counts = Counter(byte_data)
        total = len(byte_data)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy


    def byte_entropy_of_array(self, arr: np.ndarray) -> float:
        return self.byte_entropy(arr.tobytes())


    def _log_run_params(self):
        self.logger.info(f"\tUnique reference of run: {self.ref_name}")
        self.logger.info(f"\tWavefunctions will be saved in directory {m.wfcdirectory}")
        self.logger.info(f"\tDFT files are in directory {m.dftdirectory}")
        self.logger.info(f"\tThis program will run in {m.npr} processors\n")

        self.logger.info(f"\tTotal number of k-points: {m.nks}")
        self.logger.info(f"\tNumber of r-points in each direction: {m.nr1} {m.nr2} {m.nr3}")
        self.logger.info(f"\tTotal number of points in real space: {m.nr}")
        self.logger.info(f"\tNumber of bands: {m.nbnd}\n")

        self.logger.info(f"\tPoint choosen for sincronizing phases:  {m.rpoint}\n")
    
    def _wfck2r(self, nk_point: int, initial_band: int, number_of_bands: int):
        tmp_dir = os.path.join(os.getcwd(), f"tmp_nk{nk_point}")
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            # Set the command to run
            self.logger.info(f"\tCalculating wfc for k-point {nk_point}")
            shell_cmd = self._get_command(nk_point, initial_band, number_of_bands, tmp_dir)

            # Runs the command
            output = subprocess.check_output(shell_cmd, shell=True, cwd=tmp_dir)

            print(len(output))
            # Converts fortran complex numbers to numpy format
            out1 = (output.decode("utf-8")
                .replace(")", "j")
                .replace(", -", "-")
                .replace(",  ", "+")
                .replace("(", "")
                )
            print('out1', type(out1))
        
        

            if m.noncolin:
                print('NONCOLINEAR')
                # puts the wavefunctions into a numpy array
                psi = np.fromstring(out1, dtype=complex, sep="\n")
                print('type psi', type(psi))
                # For each band, find the value of the wfc at the specific point rpoint (in real space)
                psi_rpoint = np.array([psi[int(m.rpoint) + m.nr * i] for i in range(0,2*number_of_bands,2)])

                # Calculate the phase at rpoint for all the bands
                deltaphase = np.arctan2(psi_rpoint.imag, psi_rpoint.real)

                # and the modulus of the wavefunction at the reference point rpoint (
                # will be used to verify if the wavefunction at rpoint is significantly different from zero)
                mod_rpoint = np.absolute(psi_rpoint)

                psifinal0, psifinal1 = [], []

                for i in range(0,2*number_of_bands,2):
                    self.logger.debug(f"\t{nk_point:6d}  {(int(i/2) + initial_band):4d}  {mod_rpoint[int(i/2)]:12.8f}  {deltaphase[int(i/2)]:12.8f}   {not mod_rpoint[int(i/2)] < 1e-5}")
                
                    # Subtract the reference phase for each point
                    psifinal0 += list(psi[i * m.nr : (i + 1) * m.nr] * np.exp(-1j * deltaphase[int(i/2)]))                # first part of spinor, all bands
                    psifinal1 += list(psi[m.nr + i * m.nr : m.nr + (i + 1) * m.nr] * np.exp(-1j * deltaphase[int(i/2)]))  # second part of spinor, all bands

                outfiles0 = map(lambda band: os.path.join(m.wfcdirectory, f"k0{nk_point}b0{band+initial_band}-0.wfc"), range(number_of_bands))
                outfiles1 = map(lambda band: os.path.join(m.wfcdirectory, f"k0{nk_point}b0{band+initial_band}-1.wfc"), range(number_of_bands))

                for i, outfile in enumerate(outfiles0):
                    with open(outfile, "wb") as fich:
                        np.save(fich, psifinal0[i * m.nr : (i + 1) * m.nr])
                for i, outfile in enumerate(outfiles1):
                    with open(outfile, "wb") as fich:
                        np.save(fich, psifinal1[i * m.nr : (i + 1) * m.nr])

            else:
                psi = np.fromstring(out1, dtype=complex, sep="\n")
                psi_rpoint = np.array([psi[int(m.rpoint) + m.nr * i] for i in range(number_of_bands)])
                # Calculate the phase at rpoint for all the bands
                deltaphase = np.arctan2(psi_rpoint.imag, psi_rpoint.real)

                # and the modulus of the wavefunction at the reference point rpoint (
                # will be used to verify if the wavefunction at rpoint is significantly different from zero)
                mod_rpoint = np.absolute(psi_rpoint)

                psifinal = []
            
                for i in range(number_of_bands):
                    self.logger.debug(f"\t{nk_point:6d}  {(i + initial_band):4d}  {mod_rpoint[i]:12.8f}  {deltaphase[i]:12.8f}   {not mod_rpoint[i] < 1e-5}")
                
                    # Subtract the reference phase for each point
                    psifinal += list(psi[i * m.nr : (i + 1) * m.nr] * np.exp(-1j * deltaphase[i]))
                
                psifinal = np.array(psifinal)
        #        print('type psifinal', psifinal.dtype)
        #        self.analyze_psi_entropy(psifinal)
                outfiles = map(lambda band: os.path.join(m.wfcdirectory, f"k0{nk_point}b0{band+initial_band}.wfc"), range(number_of_bands))
            
                for i, outfile in enumerate(outfiles):
                    with open(outfile, "wb") as fich:
                        np.save(fich, psifinal[i * m.nr : (i + 1) * m.nr])

        finally:
                # GUARANTEED cleanup
            try:
                import shutil
                shutil.rmtree(tmp_dir)
            except Exception as e:
                self.logger.warning(f"Failed to remove {tmp_dir}: {e}")



    def threshold_array(self, arr, thresh, rel):
        if rel:
            norm = np.max(np.abs(arr))
            thresh = thresh * norm
        out = arr.copy()
        out[np.abs(out) < thresh] = 0.0
        return out



    def trunc_mantissa_float64_array(self, arr, keep_bits):
        """Keep only the top `keep_bits` mantissa bits in a float64 array."""
        a = np.ascontiguousarray(arr, dtype=np.float64)
        print(a.dtype)
        bits = a.view(np.uint64)
        mantissa_mask = np.uint64((1 << 52) - 1)
        exponent_and_sign = bits & (~mantissa_mask)
        mantissa = bits & mantissa_mask
        shift = 52 - keep_bits
        mantissa = (mantissa >> shift) << shift
        new_bits = exponent_and_sign | mantissa
        return new_bits.view(np.float64).reshape(arr.shape)



    # -------------------------
    # If psi is complex, analyze real and imag separately and report average
    def analyze_psi_entropy(self, psi: np.ndarray):
        print("=== Byte-level entropy ===")
        raw_bytes_entropy = self.byte_entropy_of_array(psi)
        print(f"Entropy of psi.tobytes(): {raw_bytes_entropy:.3f} bits/byte (max=8)")
        predicted_cr = 8.0 / raw_bytes_entropy if raw_bytes_entropy > 0 else float('inf')
        print(f"Predicted maximum lossless CR ≈ {predicted_cr:.3f}x")



    def _get_command(self, nk_point, initial_band, number_of_bands, tmp_dir):
#        mpi = "" if m.npr == 1 else f"mpirun -np {m.npr} "
        mpi = ""
        wfck_output_file = os.path.join(tmp_dir, "wfck2r.oct")
    
        command = f"""&inputpp prefix = '{m.prefix}',\
                    outdir = '{m.outdir}',\
                    first_k = {nk_point + 1},\
                    last_k = {nk_point + 1},\
                    first_band = {initial_band + 1},\
                    last_band = {initial_band + number_of_bands},\
                    loctave = .true., /"""
    
        if m.noncolin:
            cmd = f'echo "{command}" | {mpi} wfck2rFR.x > {tmp_dir}/tmp; tail -{m.nr * number_of_bands * 2} {wfck_output_file}'
        else:
            cmd = f'echo "{command}" | {mpi} wfck2r.x > {tmp_dir}/tmp; tail -{m.nr * number_of_bands} {wfck_output_file}'
    
        return cmd

     #   p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(f)
    #WfcGenerator().run()
#MPI.Finalize()

