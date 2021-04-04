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
sensor_1_left_start = 2.5 # um
sensor_1_left_end   = 3.0 # um

sensor_1_right_start = 2.5 # um
sensor_1_right_end   = 3 # um

sensor_1_left_indices  = (y[0, :] > sensor_1_left_start ) & (y[0, :] < sensor_1_left_end)
sensor_1_right_indices = (y[-1, :] > sensor_1_right_start) & (y[-1, :] < sensor_1_right_end)


time         = np.loadtxt("data/time.txt")
edge_density = np.loadtxt("data/edge_density.txt")
q2           = np.loadtxt("data/q2_edge.txt")


print (time.shape)
print (edge_density.shape)
print (q2.shape)

N_spatial = edge_density.shape[1]
time_indices = time > 25

from scipy.signal import argrelmax

AC_freq = 1./100
pl.plot(time, np.sin(2*np.pi*AC_freq*time), color = 'k', label = "Drive")
for index in [0, int(N_spatial/2), N_spatial-1]:
    index = index
    norm = np.max(edge_density[time_indices, index]) - 1.*np.mean(edge_density[time_indices, index])
    pl.plot(time[time_indices], (edge_density[time_indices, index] - 1.*np.mean(edge_density[time_indices, index]))/norm, label = "y = %.1f $\mu$m"%q2[index])
    
    max_indices = argrelmax(edge_density[time_indices, index])[0]
    print (max_indices)
    #print (time[time_indices][max_indices[1]] - time[time_indices][max_indices[0]])
    #for idx in max_indices:
    #    print (idx)
    #    pl.axvline(time[time_indices][idx])    

    pl.xlim([0, 400])

pl.legend(loc = 'best')
pl.savefig("images/iv.png")
