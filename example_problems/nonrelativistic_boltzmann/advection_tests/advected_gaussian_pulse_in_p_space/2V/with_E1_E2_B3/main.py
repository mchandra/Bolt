import arrayfire as af
import numpy as np
import h5py

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.BGK,
                         moments
                        )

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)

# Time parameters:
dt      = 0.001
t_final = 0.5

time_array  = np.arange(0, t_final + dt, dt)

# Storing data at time t = 0:
h5f = h5py.File('dump/0000.h5', 'w')
h5f.create_dataset('distribution_function', data = nls.f)
h5f.create_dataset('p1', data = nls.p1_center)
h5f.create_dataset('p2', data = nls.p2_center)
h5f.close()

f_initial = nls.f.copy()

for time_index, t0 in enumerate(time_array[1:]):

    nls.strang_timestep(dt)

    print(af.mean(af.abs(nls.f - f_initial)))
    
    h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'w')
    h5f.create_dataset('distribution_function', data = nls.f)
    h5f.create_dataset('p1', data = nls.p1_center)
    h5f.create_dataset('p2', data = nls.p2_center)
    h5f.close()
