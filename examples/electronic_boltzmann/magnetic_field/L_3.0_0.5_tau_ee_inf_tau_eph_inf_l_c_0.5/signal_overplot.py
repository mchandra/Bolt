import arrayfire as af
import numpy as np
from scipy.signal import correlate
import glob
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
matplotlib.use('agg')
import pylab as pl
#import yt
#yt.enable_parallelism()
import os

import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc

import PetscBinaryIO

import domain
import boundary_conditions
import params
import initialize


# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 4
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 25
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'

N_q1 = domain.N_q1
N_q2 = domain.N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

source_start = params.contact_start
source_end   = params.contact_end

drain_start  = params.contact_start
drain_end    = params.contact_end

source_indices =  (q2 > source_start) & (q2 < source_end)
drain_indices  =  (q2 > drain_start)  & (q2 < drain_end )

# Left needs to be near source, right sensor near drain
sensor_1_left_start = 8.5 # um
sensor_1_left_end   = 9.5 # um

sensor_1_right_start = 8.5 # um
sensor_1_right_end   = 9.5 # um

sensor_1_left_indices  = (q2 > sensor_1_left_start ) & (q2 < sensor_1_left_end)
sensor_1_right_indices = (q2 > sensor_1_right_start) & (q2 < sensor_1_right_end)

sensor_2_left_start = 6.5 # um
sensor_2_left_end   = 7.5 # um

sensor_2_right_start = 6.5 # um
sensor_2_right_end   = 7.5 # um

sensor_2_left_indices  = (q2 > sensor_2_left_start ) & (q2 < sensor_2_left_end)
sensor_2_right_indices = (q2 > sensor_2_right_start) & (q2 < sensor_2_right_end)

io = PetscBinaryIO.PetscBinaryIO()

filepath = os.getcwd()
moment_files 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))

filepath = os.getcwd() + '/../amp_geom_without_base_hydro'
moment_files_2 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files_2 = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))

filepath = os.getcwd() + '/../amp_geom_without_base_hydro_l_mc_0.001'
moment_files_3 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files_3 = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))

dt = params.dt

#time_array = params.dt_dump_moments * np.arange(0, moment_files.size, 1)
time_array = params.dt_dump_moments*np.arange(0, moment_files.size, 1)

file_number = 0
moments = io.readBinaryFile(moment_files[file_number])
moments = moments[0].reshape(N_q2, N_q1, 3)

density_bg = moments[:, :, 0]


dx = q1[1] - q1[0]
dy = q2[1] - q2[0]


print("Reading sensor signal...")
for file_number, dump_file in enumerate(moment_files):
    file_number = -1
    print ("File number : ", file_number)
    moments = io.readBinaryFile(moment_files[file_number])
    moments = moments[0].reshape(N_q2, N_q1, 3)
    
    density = moments[:, :, 0]
    density = density - density_bg
    density = density - np.mean(density[:, -1])
    
    moments = io.readBinaryFile(moment_files_2[file_number])
    moments = moments[0].reshape(N_q2, N_q1, 3)
    
    density_2 = moments[:, :, 0]
    density_2 = density_2 - density_bg
    density_2 = density_2 - np.mean(density_2[:, -1])
    
    moments = io.readBinaryFile(moment_files_3[file_number])
    moments = moments[0].reshape(N_q2, N_q1, 3)
    
    density_3 = moments[:, :, 0]
    density_3 = density_3 - density_bg
    density_3 = density_3 - np.mean(density_3[:, -1])
    
    pl.plot(q1, density[-1, :], color = 'C0', lw = 3)
    pl.plot(q1, density_2[-1, :], color = 'C1', lw = 3)
    pl.plot(q1, density_3[-1, :], color = 'C2', lw = 3)
    pl.ylim(ymin = -10, ymax = 2)

    #pl.axvline(4.85, color = 'k', ls = '--')
    #pl.axvline(3.85, color = 'k', ls = '--')
    pl.axhline(0., color = 'k', ls = '--')
    #pl.axvline(4.5, color = 'k', ls = '--')
    #pl.axvspan(4.8, 4.9, color = 'k', alpha = 0.5)
    #pl.axvspan(0.1, 0.2, color = 'k', alpha = 0.5)

    pl.xlabel("x ($\mu$m)")
    #pl.xlabel("y ($\mu$m)")
    pl.ylabel("Voltage")


    pl.tight_layout()
    pl.savefig('images/iv' + '.png')
    pl.clf()
    

