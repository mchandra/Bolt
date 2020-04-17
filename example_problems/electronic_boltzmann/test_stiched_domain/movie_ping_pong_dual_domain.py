import arrayfire as af
import numpy as np
from scipy.signal import correlate
import glob
import os
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib import colors
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


class MidpointNormalize (colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
    # I'm ignoring masked values and all kinds of edge cases to make
    # a simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))



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
N_p3 = domain.N_p3

p1 = domain.p1_start[0] + (0.5 + np.arange(N_p1)) * (domain.p1_end[0] - \
        domain.p1_start[0])/N_p1
p2 = domain.p2_start[0] + (0.5 + np.arange(N_p2)) * (domain.p2_end[0] - \
        domain.p2_start[0])/N_p2

N_s = 1

filepath = os.getcwd()

distribution_function_files_1 =  np.sort(glob.glob(filepath+'/dump_f_1/*.bin'))
distribution_function_files_2 =  np.sort(glob.glob(filepath+'/dump_f_2/*.bin'))

time_array = np.loadtxt("dump_time_array.txt")

io = PetscBinaryIO.PetscBinaryIO()

for file_number, dump_file in enumerate(distribution_function_files_1):

    print("file number = ", file_number, "of ", distribution_function_files_1.size)
    
    dist_func_file = distribution_function_files_1[file_number]
    dist_func_1 = io.readBinaryFile(dist_func_file)
    dist_func_1 = dist_func_1[0].reshape(N_q2_1, N_q1_1, N_s, N_p3, N_p2, N_p1) 
    
    dist_func_file = distribution_function_files_2[file_number]
    dist_func_2 = io.readBinaryFile(dist_func_file)
    dist_func_2 = dist_func_2[0].reshape(N_q2_2, N_q1_2, N_s, N_p3, N_p2, N_p1) 
    
    dist_func_1_integral_over_p = np.mean(dist_func_1, axis = (2, 3, 4, 5))
    dist_func_2_integral_over_p = np.mean(dist_func_2, axis = (2, 3, 4, 5))

    print (dist_func_1_integral_over_p.shape)
    print (dist_func_2_integral_over_p.shape)

    #dist_func_integral_over_p = np.concatenate((dist_func_1_integral_over_p, dist_func_2_integral_over_p), axis = 0)
    #print (dist_func_integral_over_p.shape)

    #dist_func_integral_over_p = dist_func_integral_over_p - np.mean(dist_func_integral_over_p)
    #f_min = np.min(dist_func_integral_over_p)
    #f_max = np.max(dist_func_integral_over_p)

    pl.subplot(211)
    #pl.contourf(q1_meshgrid, q2_meshgrid, dist_func_integral_over_p.transpose(), 20,
    #    norm = MidpointNormalize(midpoint=0, vmin=f_min, vmax=f_max), cmap='bwr')
    pl.contourf(q1_meshgrid_1, q2_meshgrid_1, dist_func_1_integral_over_p.transpose(), 20, cmap='bwr')
    
    pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
        
    pl.gca().set_aspect('equal')
    #pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    pl.ylabel(r'$y\;(\mu \mathrm{m})$')
    
    pl.subplot(212)
    #pl.contourf(q1_meshgrid, q2_meshgrid, dist_func_integral_over_p.transpose(), 20,
    #    norm = MidpointNormalize(midpoint=0, vmin=f_min, vmax=f_max), cmap='bwr')
    pl.contourf(q1_meshgrid_2, q2_meshgrid_2, dist_func_2_integral_over_p.transpose(), 20, cmap='bwr')
    
    #pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
        
    pl.gca().set_aspect('equal')
    pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    pl.ylabel(r'$y\;(\mu \mathrm{m})$')
    
    pl.savefig('images/dump_%06d'%file_number + '.png')
    pl.clf()
    

