#import arrayfire as af
import numpy as np
from scipy.signal import correlate
import glob
import h5py
import os
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib import transforms, colors
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
matplotlib.use('agg')
import pylab as pl
#import yt
#yt.enable_parallelism()

#import petsc4py, sys; petsc4py.init(sys.argv)
#from petsc4py import PETSc
import PetscBinaryIO

#import domain

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

class MidpointNormalize (colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
    # I'm ignoring masked values and all kinds of edge cases to make
    # a simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

io = PetscBinaryIO.PetscBinaryIO()


N_q1 = 30#domain.N_q1
N_q2 = 120#domain.N_q2

q1_start = 0.#domain.q1_start
q1_end   = 1.25#domain.q1_end
q2_start = 0.#domain.q2_start
q2_end   = 5.#domain.q2_end

q1 = q1_start + (0.5 + np.arange(N_q1)) * (q1_end - q1_start)/N_q1
q2 = q2_start + (0.5 + np.arange(N_q2)) * (q2_end - q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

coords = io.readBinaryFile("coords.bin")
coords = coords[0].reshape(N_q2, N_q1, 21)
    
x = coords[:, :, 0].T
y = coords[:, :, 1].T



N_p1 = 32#domain.N_p1
N_p2 = 512#domain.N_p2

p1_start = [0.0054]#domain.p1_start
p1_end   = [0.034199999999999994]#domain.p1_end
p2_start = [-np.pi]#domain.p2_start
p2_end   = [np.pi]#domain.p2_end

p1 = p1_start[0] + (0.5 + np.arange(N_p1)) * (p1_end[0] - p1_start[0])/N_p1
p2 = p2_start[0] + (0.5 + np.arange(N_p2)) * (p2_end[0] - p2_start[0])/N_p2

p1_meshgrid, p2_meshgrid = np.meshgrid(p1, p2)

p_x = p1_meshgrid * np.cos(p2_meshgrid)
p_y = p1_meshgrid * np.sin(p2_meshgrid)

#p2_meshgrid, p1_meshgrid = np.meshgrid(p2, p1)


filepath = os.getcwd()
moment_files 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))
dist_func_files 		  = np.sort(glob.glob(filepath+'/dump_f/*.bin'))

dist_func_bg_file = dist_func_files[0]
dist_func_file = dist_func_files[-1]

dist_func_background = io.readBinaryFile(dist_func_bg_file)
#dist_func_background = dist_func_background[0].reshape(N_q2, N_q1, N_p2, N_p1)
dist_func_background = dist_func_background[0].reshape(N_q1, N_q2, 1, 1, N_p2, N_p1)
dist_func = io.readBinaryFile(dist_func_file)

print (dist_func[0].shape)

dist_func = dist_func[0].reshape(N_q1, N_q2, 1, 1, N_p2, N_p1)


N = 7
for index_1 in range(N):
    for index_2 in range(N):

        q1_position = int(N_q1*((index_1/N)+(1/(2*N))))
        q2_position = int(N_q2*((index_2/N)+(1/(2*N))))
        
        #a = np.max((dist_func - dist_func_background)[q2_position, q1_position, :, :])
        #b = np.abs(np.min((dist_func - dist_func_background)[q2_position, q1_position, :, :]))
        #norm_factor = np.maximum(a, b)
        #f_at_desired_q = \
        #        np.reshape((dist_func-dist_func_background)[q2_position, q1_position, :, :],
        #        [N_p2, N_p1])/norm_factor
        
        f_at_desired_q = np.reshape((dist_func - \
            dist_func_background)[q1_position, q2_position, :],
                                [N_p2, N_p1]
                               )
        density_max = np.max(f_at_desired_q)
        density_min = np.min(f_at_desired_q)

        print(index_1, index_2, density_max, density_min)

        #pl.contourf(p1_meshgrid, p2_meshgrid, f_at_desired_q, 100, cmap='bwr')

        #np.savetxt('data/f_vs_theta_%d_%d.txt'%(index_1, index_2), f_at_desired_q)
        #f = np.loadtxt('data/f_vs_theta_%d_%d.txt'%(index_1, index_2))
    
    
        #pl.contourf(p_x, p_y, f_at_desired_q, 100, cmap='bwr')
        pl.contourf(p_x, p_y, f_at_desired_q, 100, norm=MidpointNormalize(midpoint=0, vmin=density_min, vmax=density_max), cmap='bwr')
        plot_grid(p_x[::1, ::1], p_y[::1, ::1], alpha=0.2)
        #pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
        pl.xlabel('$p_x$')
        pl.ylabel('$p_y$')
        pl.gca().set_aspect('equal')
        pl.savefig('images/dist_func_at_a_point_%d_%d.png'%(index_1, index_2))       
        pl.clf()

