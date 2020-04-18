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

def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or pl.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()

N_q1 = domain.N_q1
N_q2 = domain.N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

#X = q1_meshgrid
#Y = q2_meshgrid

#q1_meshgrid = af.from_ndarray(q1_meshgrid)
#q2_meshgrid = af.from_ndarray(q2_meshgrid)
#
#x, y = coords.get_cartesian_coords_for_post(q1_meshgrid, q2_meshgrid)
#
#x = x.to_ndarray()
#y = y.to_ndarray()

coords = io.readBinaryFile("coords.bin")
coords = coords[0].reshape(N_q2, N_q1, 7)
    
x = coords[:, :, 0].T
y = coords[:, :, 1].T


N_p1 = domain.N_p1
N_p2 = domain.N_p2

p1 = domain.p1_start[0] + (0.5 + np.arange(N_p1)) * (domain.p1_end[0] - \
        domain.p1_start[0])/N_p1
p2 = domain.p2_start[0] + (0.5 + np.arange(N_p2)) * (domain.p2_end[0] - \
        domain.p2_start[0])/N_p2

print ('Momentum space : ', p1[-1], p2[int(N_p2/2)])

filepath = os.getcwd()# + '/backup'
moment_files 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))

print ("moment files : ", moment_files.size)
print ("lagrange multiplier files : ", lagrange_multiplier_files.size)

dt = 0.025/8#params.dt
#dump_interval = params.dump_steps

time_array = np.loadtxt(filepath+"/dump_time_array.txt")


for file_number, dump_file in enumerate(moment_files[:]):

    file_number = -1
    print("file number = ", file_number, "of ", moment_files.size)

    moments = io.readBinaryFile(moment_files[file_number])
    moments = moments[0].reshape(N_q2, N_q1, 3)
    
    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]

    lagrange_multipliers = \
        io.readBinaryFile(lagrange_multiplier_files[file_number])
    lagrange_multipliers = lagrange_multipliers[0].reshape(N_q2, N_q1, 7)
    
    mu           = lagrange_multipliers[:, :, 0]
    mu_ee        = lagrange_multipliers[:, :, 1]
    T_ee         = lagrange_multipliers[:, :, 2]
    vel_drift_x  = lagrange_multipliers[:, :, 3]
    vel_drift_y  = lagrange_multipliers[:, :, 4]

    print ("x.shape : ", x.shape)    

    plot_grid(x[::5, ::5], y[::5, ::5], alpha=0.5)
    #pl.contourf(x, y, density.T, 100, cmap='bwr')
    pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
    #pl.colorbar()
    

#    # Plot streamlines for domains 3 and 9
#    i_start = 188
#    i_end   = 605-142-29-27
#    d = 5*(i_end - i_start)/N_q1
#    print (d)
#    pl.streamplot(x[i_start:i_end, 0], y[i_start, :], 
#                  vel_drift_x[:, i_start:i_end], vel_drift_y[:, i_start:i_end],
#                  density=d, color='k',
#                  linewidth=0.7, arrowsize=1
#                 )
#
#    # Plot streamlines for domains 1 and 7
#    i_start = 0
#    i_end   = 150
#    d = 5*(i_end - i_start)/N_q1
#    print (d)
#    pl.streamplot(x[i_start:i_end, 0], y[i_start, :], 
#                  vel_drift_x[:, i_start:i_end], vel_drift_y[:, i_start:i_end],
#                  density=d, color='k',
#                  linewidth=0.7, arrowsize=1
#                 )
#
#    # Plot streamlines for domains 6 and 12
#    i_start = 605-142
#    i_end   = 605
#    d = 5*(i_end - i_start)/N_q1
#    print (d)
#    pl.streamplot(x[i_start:i_end, 0], y[i_start, :100], 
#                  vel_drift_x[:100, i_start:i_end], vel_drift_y[:100, i_start:i_end],
#                  density=0.58*d, color='k',
#                  linewidth=0.7, arrowsize=1
#                 )
#    pl.streamplot(x[i_start:i_end, 0], y[i_start, 100:], 
#                  vel_drift_x[100:, i_start:i_end], vel_drift_y[100:, i_start:i_end],
#                  density=0.4*d, color='k',
#                  linewidth=0.7, arrowsize=1
#                 )
#
#
#    # Plot streamlines for domain 4
#    i_start = 605-142-29-27
#    i_end   = 605-142-29
#    d = 2*5*(i_end - i_start)/N_q1
#    print (d)
#    pl.streamplot(x[i_start:i_end, 0], y[i_start, :100], 
#                  vel_drift_x[:100, i_start:i_end], vel_drift_y[:100, i_start:i_end],
#                  density=d, color='k',
#                  linewidth=0.7, arrowsize=1
#                 )
    
    #pl.xlim([-1, 27])
    #pl.ylim([q2[0], q2[-1]])
    
    pl.gca().set_aspect('equal')
    pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    pl.ylabel(r'$y\;(\mu \mathrm{m})$')
    pl.suptitle('$\\tau_\mathrm{mc} = \infty$, $\\tau_\mathrm{mr} = \infty$')
    pl.savefig('images/dump_' + '%06d'%file_number + '.png')
    pl.clf()

