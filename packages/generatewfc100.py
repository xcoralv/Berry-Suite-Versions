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
import math
import bz2
from tempfile import TemporaryDirectory

try:
    import berry._subroutines.loadmeta as m
    import berry._subroutines.loaddata as d
    from berry.compression_utils1 import truncate_complex, save_bz2_npy, CompressionTimer
    from berry.compression_utils2 import save_array, load_array
except:
    pass

timer = CompressionTimer()

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
        self.logger.header()
        self._log_run_params()

        if m.noncolin:
            self.k2r_program = "wfck2rFR.x"
            self.logger.info("\tNoncolinear calculation, will use wfck2rFR.x")
        else:
            self.k2r_program = "wfck2r.x"
            self.logger.info("\tNonrelativistic calculation, will use wfck2r.x")

        if isinstance(self.nk_points, range):
            self.logger.info("\n\tWill run for all k-points and bands")
            self.logger.info(f"\tThere are {m.nks} k-points and {m.nbnd} bands.\n")
            print("Task list:", [(nk, 0, m.nbnd) for nk in self.nk_points])
            with (processes = self.npr) as pool:
                results = pool.starmap(self._wfck2r, [(nk, 0, m.nbnd) for nk in self.nk_points])
                total_compression_time = sum(results)
        else:
            if isinstance(self.bands, range):
                self.logger.info(f"\tWill run for k-point {self.nk_points} and all bands")
                self.logger.info(f"\tThere are {m.nks} k-points and {m.nbnd} bands.\n")
                self._wfck2r(self.nk_points, 0, m.nbnd)
            else:
                self.logger.info(f"\tWill run just for k-point {self.nk_points} and band {self.bands}.\n")
                self._wfck2r(self.nk_points, self.bands, 1)

        self.logger.info("\n\tRemoving temporary file 'tmp'")
#        os.system(f"rm {os.getcwd()}/tmp")
        self.logger.info(f"\tRemoving quantum expresso output file '{m.wfck2r}'")
#        os.system(f"rm {os.path.join(os.getcwd(),m.wfck2r)}")
        print('total comp time:', total_compression_time)
        self.logger.footer()


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



    def _wfck2r(self, nk_point: int, initial_band: int, number_of_bands: int) -> float:
        """
        Returns total compression time used in this call.
        """
        total_compression_time = 0.0
        tmp_dir = os.path.join(os.getcwd(), f"tmp_nk{nk_point}")
        os.makedirs(tmp_dir, exist_ok=True)
        #with TemporaryDirectory(prefix=f"tmp_nk{nk_point}_", dir=os.getcwd()) as tmp_dir:
        try:
            self.logger.info(f"\tCalculating wfc for k-point {nk_point}")
            #tmp_dir = f"{os.getcwd()}/tmp_nk{nk_point}"
            #os.makedirs(tmp_dir, exist_ok=True)

            shell_cmd = self._get_command(nk_point, initial_band, number_of_bands, tmp_dir)
            output = subprocess.check_output(shell_cmd, shell=True, cwd=tmp_dir)

            # Convert Fortran output to numpy complex array
            out1 = output.decode("utf-8").replace(")", "j").replace(", -", "-").replace(",  ", "+").replace("(", "")
            psi = np.fromstring(out1, dtype=complex, sep="\n")

            if m.noncolin:
                # Non-colinear case
                psifinal0, psifinal1 = [], []
                psi_rpoint = np.array([psi[int(m.rpoint) + m.nr * i] for i in range(0, 2 * number_of_bands, 2)])
                deltaphase = np.arctan2(psi_rpoint.imag, psi_rpoint.real)
                for i in range(0, 2 * number_of_bands, 2):
                    psifinal0 += list(psi[i * m.nr: (i + 1) * m.nr] * np.exp(-1j * deltaphase[int(i/2)]))
                    psifinal1 += list(psi[m.nr + i * m.nr: m.nr + (i + 1) * m.nr] * np.exp(-1j * deltaphase[int(i/2)]))
                psifinal0 = truncate_complex(np.array(psifinal0))
                psifinal1 = truncate_complex(np.array(psifinal1))

                outfiles0 = [os.path.join(m.wfcdirectory, f"k0{nk_point}b0{band+initial_band}-0.wfc.bz2") for band in range(number_of_bands)]
                outfiles1 = [os.path.join(m.wfcdirectory, f"k0{nk_point}b0{band+initial_band}-1.wfc.bz2") for band in range(number_of_bands)]

                for i, outfile in enumerate(outfiles0):
                    dt = save_bz2_npy(outfile, psifinal0[i * m.nr: (i + 1) * m.nr])
                    total_compression_time += dt
                for i, outfile in enumerate(outfiles1):
                    dt = save_bz2_npy(outfile, psifinal1[i * m.nr: (i + 1) * m.nr])
                    total_compression_time += dt

            else:
                # Non-relativistic case
                psifinal = []
                psi_rpoint = np.array([psi[int(m.rpoint) + m.nr * i] for i in range(number_of_bands)])
                deltaphase = np.arctan2(psi_rpoint.imag, psi_rpoint.real)
                for i in range(number_of_bands):
                    psifinal += list(psi[i * m.nr: (i + 1) * m.nr] * np.exp(-1j * deltaphase[i]))
                psifinal = truncate_complex(np.array(psifinal))

                outfiles = [os.path.join(m.wfcdirectory, f"k0{nk_point}b0{band+initial_band}.wfc") for band in range(number_of_bands)]
                for i, outfile in enumerate(outfiles):
                    dt = save_array(outfile, psifinal[i * m.nr: (i + 1) * m.nr])
                    total_compression_time += dt

        finally:
            # GUARANTEED cleanup
            try:
                import shutil
                shutil.rmtree(tmp_dir)
            except Exception as e:
                self.logger.warning(f"Failed to remove {tmp_dir}: {e}")

        return total_compression_time

    def _get_command(self, nk_point, initial_band, number_of_bands, tmp_dir):
#    def _get_command(self, nk_point: int, initial_band: int, number_of_bands: int, tmp_dir):
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


