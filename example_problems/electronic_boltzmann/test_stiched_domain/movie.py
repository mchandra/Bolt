import arrayfire as af
import numpy as np
from scipy.signal import correlate
import glob
import os
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

# Common domain stuff stored here
import domain
import params

import domain_1
import params_1
import coords_1

import domain_2
import params_2
import coords_2


# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 8, 8
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

def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or pl.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


N_q1_1 = domain_1.N_q1
N_q2_1 = domain_1.N_q2

q1_1 = domain_1.q1_start + (0.5 + np.arange(N_q1_1)) * (domain_1.q1_end - \
        domain_1.q1_start)/N_q1_1
q2_1 = domain_1.q2_start + (0.5 + np.arange(N_q2_1)) * (domain_1.q2_end - \
        domain_1.q2_start)/N_q2_1

dq1_1 = (domain_1.q1_end - domain_1.q1_start)/N_q1_1
dq2_1 = (domain_1.q2_end - domain_1.q2_start)/N_q2_1

q2_meshgrid_1, q1_meshgrid_1 = np.meshgrid(q2_1, q1_1)


q1_meshgrid_1 = af.from_ndarray(q1_meshgrid_1)
q2_meshgrid_1 = af.from_ndarray(q2_meshgrid_1)

x, y = coords_1.get_cartesian_coords(q1_meshgrid_1, q2_meshgrid_1)

x_1 = x.to_ndarray()
y_1 = y.to_ndarray()



N_q1_2 = domain_2.N_q1
N_q2_2 = domain_2.N_q2

q1_2 = domain_2.q1_start + (0.5 + np.arange(N_q1_2)) * (domain_2.q1_end - \
        domain_2.q1_start)/N_q1_2
q2_2 = domain_2.q2_start + (0.5 + np.arange(N_q2_2)) * (domain_2.q2_end - \
        domain_2.q2_start)/N_q2_2

dq1_2 = (domain_2.q1_end - domain_1.q1_start)/N_q1_1
dq2_2 = (domain_2.q2_end - domain_1.q2_start)/N_q2_1

q2_meshgrid_2, q1_meshgrid_2 = np.meshgrid(q2_2, q1_2)

q1_meshgrid_2 = af.from_ndarray(q1_meshgrid_2)
q2_meshgrid_2 = af.from_ndarray(q2_meshgrid_2)

x, y = coords_2.get_cartesian_coords(q1_meshgrid_2, q2_meshgrid_2)

x_2 = x.to_ndarray()
y_2 = y.to_ndarray()


# Stich domains
#N_q1 = N_q1_1
#N_q2 = N_q2_1 + N_q2_2

#q1_start = 0.
#q1_end   = domain_1.q1_end
#q2_start = 0.
#q2_end   = domain_1.q2_end + domain_2.q2_end

#dq1 = (q1_end - q1_start)/N_q1
#dq2 = (q2_end - q2_start)/N_q2

#q1 = q1_start + (0.5 + np.arange(N_q1)) * dq1
#q2 = q2_start + (0.5 + np.arange(N_q2)) * dq2

#q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

N_p1 = domain.N_p1
N_p2 = domain.N_p2

p1 = domain.p1_start[0] + (0.5 + np.arange(N_p1)) * (domain.p1_end[0] - \
        domain.p1_start[0])/N_p1
p2 = domain.p2_start[0] + (0.5 + np.arange(N_p2)) * (domain.p2_end[0] - \
        domain.p2_start[0])/N_p2

print ('Momentum space : ', p1[-1], p2[int(N_p2/2)])

filepath = os.getcwd()
moment_files_1 		  = np.sort(glob.glob(filepath+'/dump_moments_1/*.bin'))
moment_files_2 		  = np.sort(glob.glob(filepath+'/dump_moments_2/*.bin'))


time_array = np.loadtxt("dump_time_array.txt")

io = PetscBinaryIO.PetscBinaryIO()

for file_number, dump_file in enumerate(moment_files_1[::-1]):

    file_number = -1
    print("file number = ", file_number, "of ", moment_files_1.size)

    moments = io.readBinaryFile(moment_files_1[file_number])
    moments_1 = moments[0].reshape(N_q2_1, N_q1_1, 3)
    
    density_1 = moments_1[:, :, 0]
    
    moments = io.readBinaryFile(moment_files_2[file_number])
    moments_2 = moments[0].reshape(N_q2_2, N_q1_2, 3)
    
    density_2 = moments_2[:, :, 0]


    #pl.subplot(212)
    plot_grid(x_1[::5, ::5], y_1[::5, ::5], alpha=0.5)
    pl.contourf(x_1, y_1, density_1.T, 100, cmap='bwr')
    #pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
    
    plot_grid(x_2[::5, ::5], y_2[::5, ::5], alpha=0.5)
    pl.contourf(x_2, y_2, density_2.T, 100, cmap='bwr')
    
    pl.colorbar()
    pl.gca().set_aspect('equal')
    #pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    #pl.ylabel(r'$y\;(\mu \mathrm{m})$')
 
    #pl.suptitle('$\\tau_\mathrm{mc} = \infty$, $\\tau_\mathrm{mr} = \infty$')
    pl.savefig('images/dump_' + '%06d'%file_number + '.png')
    pl.clf()

