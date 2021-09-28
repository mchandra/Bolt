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

N_q1 = domain.N_q1
N_q2 = domain.N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

source_start = params.contact_start
source_end   = params.contact_end

drain_start  = params.contact_start
drain_end    = params.contact_end

source_indices =  (q2 > source_start) & (q2 < source_end)
drain_indices  =  (q2 > drain_start)  & (q2 < drain_end )

# Left needs to be near source, right sensor near drain
sensor_1_left_start = 8.5 # um
sensor_1_left_end   = 9.5 # um

sensor_1_right_start = 8.5 # um
sensor_1_right_end   = 9.5 # um

sensor_1_left_indices  = (q2 > sensor_1_left_start ) & (q2 < sensor_1_left_end)
sensor_1_right_indices = (q2 > sensor_1_right_start) & (q2 < sensor_1_right_end)

sensor_2_left_start = 6.5 # um
sensor_2_left_end   = 7.5 # um

sensor_2_right_start = 6.5 # um
sensor_2_right_end   = 7.5 # um

sensor_2_left_indices  = (q2 > sensor_2_left_start ) & (q2 < sensor_2_left_end)
sensor_2_right_indices = (q2 > sensor_2_right_start) & (q2 < sensor_2_right_end)

io = PetscBinaryIO.PetscBinaryIO()

filepath = os.getcwd()
#filepath = '/home/mchandra/gitansh/zero_T/example_problems/electronic_boltzmann/graphene/L_1.0_1.25_tau_ee_inf_tau_eph_inf_DC'
moment_files 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))

#moment_files 		  = np.sort(glob.glob(filepath+'/dumps/moments*.bin'))
#lagrange_multiplier_files = \
#        np.sort(glob.glob(filepath+'/dumps/lagrange_multipliers*.bin'))

dt = params.dt
dump_interval = params.dump_steps

file_number = 0
moments = io.readBinaryFile(moment_files[file_number])
moments = moments[0].reshape(N_q2, N_q1, 3)

density_bg = moments[:, :, 0]


sensor_1_signal_array = []
print("Reading sensor signal...")
file_number = -1
moments = io.readBinaryFile(moment_files[file_number])
moments = moments[0].reshape(N_q2, N_q1, 3)

density = moments[:, :, 0]
density = density - density_bg

lagrange_multipliers = \
        io.readBinaryFile(lagrange_multiplier_files[file_number])
lagrange_multipliers = lagrange_multipliers[0].reshape(N_q2, N_q1, 5)
    
mu           = lagrange_multipliers[:, :, 0]
mu_ee        = lagrange_multipliers[:, :, 1]
T_ee         = lagrange_multipliers[:, :, 2]
vel_drift_x  = lagrange_multipliers[:, :, 3]
vel_drift_y  = lagrange_multipliers[:, :, 4]

source = np.mean(density[source_indices, 0])
drain  = np.mean(density[drain_indices, -1])

#sensor_1_top     = (vel_drift_y[0,  :])
#sensor_1_bottom  = (vel_drift_y[-1, :])

sensor_1_top     = (density[0,  :])
sensor_1_bottom  = (density[-1, :])

indices = (q1>4.25)

pl.plot(q1[indices], sensor_1_top[indices])
pl.plot(q1[indices], sensor_1_bottom[indices])
#pl.axhline(0, color='black', linestyle='--')

#pl.legend(['Source $I(t)$', 'Measured $V(t)$'], loc=1)
#pl.text(135, 1.14, '$\phi : %.2f \; rad$' %phase_diff)
pl.xlabel(r'x ($\mu$m)')
#pl.xlim([0, 200])
#pl.ylim([-1.1, 1.1])

pl.title("No secondary Injection")
#pl.suptitle('$\\tau_\mathrm{mc} = 0.2$ ps, $\\tau_\mathrm{mr} = 0.5$ ps')
pl.savefig('images/iv' + '.png')
pl.clf()
    

