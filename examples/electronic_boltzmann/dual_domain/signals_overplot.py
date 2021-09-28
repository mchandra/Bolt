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

filepath = os.getcwd() + '/hydro_moments_2/hydro_moments_last_100_ps/*.bin'
moment_files 		  = np.sort(glob.glob(filepath))#+'/dump_moments/*.bin'))

print ("moment files : ", moment_files.size)

time_step = 0.025/4#params.dt
dump_step = 20*time_step#params.dt_dump_moments

#time_array = np.loadtxt(filepath+"/dump_time_array.txt")
time_array = dump_step * np.arange(0, moment_files.size, 1)

q1_index = (x[:, 0] > 0) & (x[:, 0] < 1); q2_index = -1

print ("x indices: ", x[q1_index, q2_index])

sensor_1_array = []
for file_number, dump_file in enumerate(moment_files[:]):

    #file_number = -1
    print("file number = ", file_number, "of ", moment_files.size)

    moments = io.readBinaryFile(moment_files[file_number])
    moments = moments[0].reshape(N_q2, N_q1, 3)
    
    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]

    signal  = np.mean(density[q2_index, q1_index])
    
    sensor_1_array.append(signal)

sensor_1_array = np.array(sensor_1_array)

filepath = os.getcwd() + '/hydro_moments/hydro_moments_last_100_ps/*.bin'
moment_files 		  = np.sort(glob.glob(filepath))#+'/dump_moments/*.bin'))

print ("moment files : ", moment_files.size)

sensor_2_array = []
for file_number, dump_file in enumerate(moment_files[:]):

    #file_number = -1
    print("file number = ", file_number, "of ", moment_files.size)

    moments = io.readBinaryFile(moment_files[file_number])
    moments = moments[0].reshape(N_q2, N_q1, 3)
    
    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]

    signal  = np.mean(density[q2_index, q1_index])
    
    sensor_2_array.append(signal)

sensor_2_array = np.array(sensor_2_array)


pl.plot(time_array, sensor_1_array) 
pl.plot(time_array, sensor_2_array) 

#pl.gca().set_aspect('equal')
pl.ylabel(r'$n$')
pl.xlabel(r'Time (ps)')
#pl.suptitle('$\\tau_\mathrm{mc} = \infty$, $\\tau_\mathrm{mr} = \infty$')
pl.suptitle('$\\tau_\mathrm{mc} = 0.1$, $\\tau_\mathrm{mr} = \infty$')
pl.savefig('images/iv.png')
pl.clf()

