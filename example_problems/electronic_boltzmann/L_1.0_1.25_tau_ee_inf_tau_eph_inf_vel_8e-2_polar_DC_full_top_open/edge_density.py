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


io = PetscBinaryIO.PetscBinaryIO()

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 7.5
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

source_start = params.contact_start
source_end   = params.contact_end

drain_start  = params.contact_start
drain_end    = params.contact_end

coords = io.readBinaryFile("coords.bin")
coords = coords[0].reshape(N_q2, N_q1, 21)
    
x = coords[:, :, 0].T
y = coords[:, :, 1].T


# Left needs to be near source, right sensor near drain
sensor_1_left_start = 0.25 # um
sensor_1_left_end   = 5.0 # um

sensor_1_right_start = 0.25 # um
sensor_1_right_end   = 5.0 # um

sensor_1_left_indices  = (y[0, :] > sensor_1_left_start ) & (y[0, :] < sensor_1_left_end)
sensor_1_right_indices = (y[-1, :] > sensor_1_right_start) & (y[-1, :] < sensor_1_right_end)


io = PetscBinaryIO.PetscBinaryIO()

filepath = os.getcwd()
moment_files 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))

dt = params.dt


file_number = 0
moments = io.readBinaryFile(moment_files[file_number])
moments = moments[0].reshape(N_q2, N_q1, 3)

density_bg = moments[:, :, 0]

skip_files = 0 # Corresponds to t = 149.937 ps
moment_files = moment_files[skip_files:]


#time_array = params.dt_dump_moments * np.arange(0, moment_files.size, 1)
time_array = 0. + params.dt_dump_moments*np.arange(0, moment_files.size, 1)


density = []
edge_density = []
print("Reading sensor signal...")
for file_number, dump_file in enumerate(moment_files):
    #file_number = -1
    print ("File number : ", file_number)
    moments = io.readBinaryFile(moment_files[file_number])
    moments = moments[0].reshape(N_q2, N_q1, 3)
    
    #density = moments[:, :, 0]
    #density = density - density_bg
    density.append(moments[:, :, 0])
    edge_density.append(density[file_number][sensor_1_left_indices, 0])

density = np.array(density)
edge_density = np.array(edge_density)

mean_density = np.mean(density)
max_density  = np.max(density)
min_density  = np.min(density)

np.savetxt("data/edge_density.txt", \
        edge_density - mean_density)
np.savetxt("data/q2_edge.txt", y[0, sensor_1_left_indices])
np.savetxt("data/time.txt", time_array)
    


