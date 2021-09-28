import arrayfire as af
import numpy as np
import os
#from scipy.signal import correlate
import glob
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
matplotlib.use('agg')
import pylab as pl
#import yt
#yt.enable_parallelism()

import PetscBinaryIO

import domain
import boundary_conditions
import params
import initialize
import coords

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

N_s = len(params.mass) # Number of species

N_p1 = domain.N_p1
N_p2 = domain.N_p2
N_p3 = domain.N_p3

N_q1 = domain.N_q1
N_q2 = domain.N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

q1_meshgrid = af.from_ndarray(q1_meshgrid)
q2_meshgrid = af.from_ndarray(q2_meshgrid)

x, y = coords.get_cartesian_coords(q1_meshgrid, q2_meshgrid)

theta = coords.get_theta_left(q1_meshgrid, q2_meshgrid)

print (theta.dims())

pl.plot(q2, theta)
pl.savefig("iv.png")
pl.clf()

x = x.to_ndarray()
y = y.to_ndarray()

#a = 0.1; k = np.pi
#x = q1_meshgrid + a*np.cos(k*q2_meshgrid)
#y = q2_meshgrid - a*np.sin(k*q1_meshgrid)



p2_start = domain.p2_start[0]
p2_end   = domain.p2_end[0]

p2 = p2_start + (0.5 + np.arange(N_p2)) * (p2_end - p2_start)/N_p2

filepath = os.getcwd()
distribution_function_files = np.sort(glob.glob(filepath+'/dump_f/*.bin'))

time_array = np.loadtxt("dump_time_array.txt")


theta_0_index = int(4*N_p2/8) # Direction of initial velocity
theta = p2[theta_0_index]

print("theta = ", theta)


for file_number, dump_file in enumerate(distribution_function_files[:]):
    
    print("file_number = ", file_number, "of", distribution_function_files[:].size)
    
    dist_func_file = distribution_function_files[file_number]
    dist_func = io.readBinaryFile(dist_func_file)
    dist_func = dist_func[0].reshape(N_q2, N_q1, N_s, N_p3, N_p2, N_p1)
    
    #dist_func_p_avged = np.mean(dist_func, axis = (2, 3,4,5))
    #print (dist_func_p_avged.shape)
    #pl.contourf(q1_meshgrid, q2_meshgrid, dist_func_p_avged.transpose(), 20, cmap='bwr')
    
    print ("Read dist func files")
    pl.contourf(y, x, dist_func[:, :, 0, 0, theta_0_index, 0].transpose(), 100, cmap='bwr')
    pl.gca().set_aspect('equal')
   
    pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
        
    pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    pl.ylabel(r'$y\;(\mu \mathrm{m})$')

    pl.savefig('images/dump_%06d'%file_number + '.png')
    pl.clf()



