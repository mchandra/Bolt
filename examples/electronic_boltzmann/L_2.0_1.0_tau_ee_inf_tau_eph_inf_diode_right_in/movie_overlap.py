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
import params
#import initialize
#import coords


# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 15, 7.5
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'normal'
pl.rcParams['font.size']       = 20
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = False
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

class MidpointNormalize (colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
    # I'm ignoring masked values and all kinds of edge cases to make
    # a simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

N_q1 = domain.N_q1
N_q2 = domain.N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)


N_p1 = domain.N_p1
N_p2 = domain.N_p2

p1 = domain.p1_start[0] + (0.5 + np.arange(N_p1)) * (domain.p1_end[0] - \
        domain.p1_start[0])/N_p1
p2 = domain.p2_start[0] + (0.5 + np.arange(N_p2)) * (domain.p2_end[0] - \
        domain.p2_start[0])/N_p2

print ('Momentum space : ', p1[-1], p2[int(N_p2/2)])

filepath = os.getcwd() + '/high_res_back'
moment_files 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))

print ("moment files : ", moment_files.size)
print ("lagrange multiplier files : ", lagrange_multiplier_files.size)


#time_array = np.loadtxt("dump_time_array.txt")
time_array = params.dt_dump_moments*np.arange(0, moment_files.size, 1)


file_number = 0
moments = io.readBinaryFile(moment_files[file_number])
moments = moments[0].reshape(N_q2, N_q1, 3)
    
density_bg = moments[:, :, 0]


filepath = os.getcwd() + '/low_res_back'
moment_files_2 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files_2 = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))

print ("moment files : ", moment_files.size)
print ("lagrange multiplier files : ", lagrange_multiplier_files.size)


file_number = 0
moments = io.readBinaryFile(moment_files_2[file_number])
moments = moments[0].reshape(120, 120, 3)
    
density_bg_2 = moments[:, :, 0]


coords = io.readBinaryFile("high_res_back/coords.bin")
coords = coords[0].reshape(N_q2, N_q1, 21)
    
x = coords[:, :, 0].T
y = coords[:, :, 1].T

coords = io.readBinaryFile("low_res_back/coords.bin")
coords = coords[0].reshape(120, 120, 21)
    
x_2 = coords[:, :, 0].T
y_2 = coords[:, :, 1].T


start_index = 0 # Make movie from just before restart point : 18.75 ps

for file_number, dump_file in enumerate(moment_files[:]):

#    file_number = -1
    print("file number = ", file_number, "of ", moment_files.size)

    moments = io.readBinaryFile(moment_files[start_index+file_number])
    moments = moments[0].reshape(N_q2, N_q1, 3)
    
    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]

    moments = io.readBinaryFile(moment_files_2[start_index+file_number])
    moments = moments[0].reshape(120, 120, 3)
    
    density_2 = moments[:, :, 0]
    j_x_2     = moments[:, :, 1]
    j_y_2     = moments[:, :, 2]


#    density = density - density_bg
    density_min = np.min(density)
    density_max = np.max(density)
    density_min_2 = np.min(density_2)
    density_max_2 = np.max(density_2)


    pl.suptitle(r'Time = ' + "%.2f"%(time_array[start_index+file_number]) + " ps")
    pl.subplot(121)
    pl.contourf(x, y, density.T, 100, norm=MidpointNormalize(midpoint=0, vmin=density_min, vmax=density_max), cmap='bwr')
    
    pl.xlim([q1[0], q1[-1]])
    pl.ylim([q2[0], q2[-1]])
    
    pl.gca().set_aspect('equal')
    pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    pl.ylabel(r'$y\;(\mu \mathrm{m})$')
    
    pl.subplot(122)
    pl.contourf(x_2, y_2, density_2.T, 100, norm=MidpointNormalize(midpoint=0, vmin=density_min_2, vmax=density_max_2), cmap='bwr')
    
    pl.xlim([q1[0], q1[-1]])
    pl.ylim([q2[0], q2[-1]])
    
    pl.gca().set_aspect('equal')
    pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    #pl.ylabel(r'$y\;(\mu \mathrm{m})$')
    
    
    #pl.suptitle('$\\tau_\mathrm{mc} = \infty$, $\\tau_\mathrm{mr} = \infty$')
    pl.savefig('images/dump_' + '%06d'%(start_index+file_number) + '.png')
    pl.clf()

