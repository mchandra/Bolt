print ("Execution started, loading libs...")

#import arrayfire as af
import numpy as np
from scipy.signal import correlate
import glob
import os
#import h5py
import matplotlib
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
#import boundary_conditions
#import params
#import initialize


#import petsc4py, sys; petsc4py.init(sys.argv)
#from petsc4py import PETSc

#from bolt.lib.physical_system import physical_system

#from bolt.lib.nonlinear_solver.nonlinear_solver \
#    import nonlinear_solver
#from bolt.lib.nonlinear_solver.EM_fields_solver.electrostatic \
#    import compute_electrostatic_fields

#import domain
#import boundary_conditions
#import params
#import initialize

#import bolt.src.electronic_boltzmann.advection_terms as advection_terms

#import bolt.src.electronic_boltzmann.collision_operator \
#    as collision_operator

#import bolt.src.electronic_boltzmann.moment_defs as moment_defs

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

print ("Loaded libs")

N_q1 = 72
N_q2 = 90

q1_start = 0.
q2_start = 0
q1_end   = 1.
q2_end   = 1.25

q1 = q1_start + (0.5 + np.arange(N_q1)) * (q1_end - q1_start)/N_q1
q2 = q2_start + (0.5 + np.arange(N_q2)) * (q2_end - q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

contact_start = 0.
contact_end   = 0.25

source_start = contact_start
source_end   = contact_end

drain_start  = contact_start
drain_end    = contact_end

source_indices =  (q2 > source_start) & (q2 < source_end)
drain_indices  =  (q2 > drain_start)  & (q2 < drain_end )

probe_1_indices = (q2 < source_end)


#Dump files
filepath = '.'
moment_files              = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))


print ("Moment files size : ", moment_files.size)
skip_files = 0

moment_files = moment_files[skip_files:]
lagrange_multiplier_files = lagrange_multiplier_files[skip_files:]

dt = 0.025/4
dump_interval = 10

#time_array = np.loadtxt("dump_time_array.txt")

time_array = dump_interval * dt * np.arange(0, moment_files.size, 1)

io = PetscBinaryIO.PetscBinaryIO()

left_edge_density  = []
right_edge_density = []
bottom_edge_density  = []
top_edge_density = []

for file_number, dump_file in enumerate(moment_files):

    dump_file = moment_files[file_number]
    print (dump_file)

    try :
        moments = io.readBinaryFile(dump_file)
        moments = moments[0].reshape(N_q2, N_q1, 3)

        print("file number = ", file_number, "of ", moment_files.size, "shape = ", moments.shape)

        density = moments[:, :, 0]

        left_edge_density.append(density[:, 0])
        right_edge_density.append(density[:, -1])
        bottom_edge_density.append(density[0, :])
        top_edge_density.append(density[-1, :])
    
    except (IndexError):
        none_array = np.zeros(shape=(N_q2,))
        none_array.fill(np.nan)
        #print (none_array)
        left_edge_density.append(none_array)
        right_edge_density.append(none_array)
        bottom_edge_density.append(none_array)
        top_edge_density.append(none_array)
        print ("######################Index Error handled at file number ", file_number)


voltage_left  = np.array(left_edge_density)
voltage_right = np.array(right_edge_density)
voltage_bottom  = np.array(bottom_edge_density)
voltage_top     = np.array(top_edge_density)

print("Left : ", voltage_left.shape)

np.savez_compressed("diode_left_in_vel_0.2",
        left=voltage_left, right=voltage_right)
#np.savez_compressed('current_L_1.0_5.25_tau_ee_inf_tau_eph_inf.txt',
#        current=current_left)

