import arrayfire as af
import numpy as np
from scipy.signal import correlate
import glob
import os
import sys
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib import transforms, colors
matplotlib.use('agg')
import pylab as pl
#import yt
#yt.enable_parallelism()

import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc
import PetscBinaryIO

import domain
#import boundary_conditions
#import params
#import initialize
#import coords


# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
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

io = PetscBinaryIO.PetscBinaryIO()

N_q1 = domain.N_q1
N_q2 = domain.N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

coords = io.readBinaryFile("coords.bin")
coords = coords[0].reshape(N_q2, N_q1, 13)
    
x = coords[:, :, 0].T
y = coords[:, :, 1].T


N_p1 = domain.N_p1
N_p2 = domain.N_p2

p1 = domain.p1_start[0] + (0.5 + np.arange(N_p1)) * (domain.p1_end[0] - \
        domain.p1_start[0])/N_p1
p2 = domain.p2_start[0] + (0.5 + np.arange(N_p2)) * (domain.p2_end[0] - \
        domain.p2_start[0])/N_p2

print ('Momentum space : ', p1[-1], p2[int(N_p2/2)])

q1_index = (x[:, 0] > 2.52+0.5); q2_index = 0

print ("x indices: ", x[q1_index, q2_index])


filepath = os.getcwd() + '/GaAs_hydro'
moment_files 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))


file_number = -1
print (moment_files[file_number])
print("file number = ", file_number, "of ", moment_files.size)

moments = io.readBinaryFile(moment_files[file_number])
moments = moments[0].reshape(N_q2, N_q1, 3)

density_hydro = moments[:, :, 0]


filepath = os.getcwd() + '/GaAs_ballistic'
moment_files 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))


file_number = -1
print (moment_files[file_number])
print("file number = ", file_number, "of ", moment_files.size)

moments = io.readBinaryFile(moment_files[file_number])
moments = moments[0].reshape(N_q2, N_q1, 3)

density_ballistic = moments[:, :, 0]




filepath = os.getcwd() + '/GaAs_Ohmic'
moment_files 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))


file_number = -1
print (moment_files[file_number])
print("file number = ", file_number, "of ", moment_files.size)

moments = io.readBinaryFile(moment_files[file_number])
moments = moments[0].reshape(N_q2, N_q1, 3)

density_ohmic = moments[:, :, 0]
    

# top_edge - mean(right-edge)
sensor_1_array  = density_hydro[q2_index, q1_index] - np.mean(density_hydro[:, -1])
sensor_2_array  = density_ballistic[q2_index, q1_index] - np.mean(density_ballistic[:, -1])
sensor_3_array  = density_ohmic[q2_index, q1_index] - np.mean(density_ohmic[:, -1])

np.savetxt("R_vs_L_hydro.txt", sensor_1_array)
np.savetxt("R_vs_L_ballistic.txt", sensor_2_array)
np.savetxt("R_vs_L_ohmic.txt", sensor_3_array)

np.savetxt("bottom_edge.txt", x[q1_index, q2_index])

pl.plot(x[q1_index, q2_index], sensor_1_array, label = "Hydro")
pl.plot(x[q1_index, q2_index], sensor_2_array, label = "Ballistic") 
pl.plot(x[q1_index, q2_index], sensor_3_array, label = "Ohmic") 

pl.axhline(0, color='k', ls = '--')

pl.ylim(ymax=150)
#pl.gca().set_aspect('equal')
pl.ylabel(r'$n$')
pl.xlabel(r'x ($\mu$m)')
pl.legend(loc='best')
#pl.suptitle('$\\tau_\mathrm{mc} = \infty$, $\\tau_\mathrm{mr} = \infty$')
#pl.suptitle('$\\tau_\mathrm{mc} = 0.1$, $\\tau_\mathrm{mr} = \infty$')
pl.savefig('images/iv.png')
pl.clf()

